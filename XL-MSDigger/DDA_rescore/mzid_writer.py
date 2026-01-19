import csv
import datetime
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

PSI_MS = "PSI-MS"
UNIMOD = "UNIMOD"
XLMOD = "XLMOD"

NS = {
    "mzid": "http://psidev.info/psi/pi/mzIdentML/1.3",
}

ET.register_namespace("", NS["mzid"])


def _cv_param(parent, accession, name, cv_ref, value=None, unit_accession=None, unit_name=None, unit_cv_ref=None):
    el = ET.SubElement(parent, f"{{{NS['mzid']}}}cvParam")
    el.set("accession", accession)
    el.set("name", name)
    el.set("cvRef", cv_ref)
    if value is not None:
        el.set("value", str(value))
    if unit_accession is not None:
        el.set("unitAccession", unit_accession)
    if unit_name is not None:
        el.set("unitName", unit_name)
    if unit_cv_ref is not None:
        el.set("unitCvRef", unit_cv_ref)
    return el


def _user_param(parent, name, value=None):
    el = ET.SubElement(parent, f"{{{NS['mzid']}}}userParam")
    el.set("name", name)
    if value is not None:
        el.set("value", str(value))
    return el


def parse_fasta(fasta_path):
    seqs = {}
    current_id = None
    current_desc = None
    buf = []
    with open(fasta_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(buf)
                    seqs[current_id] = (seq, current_desc)
                header = line[1:]
                current_desc = header
                parts = header.split("|")
                if len(parts) >= 3 and parts[0] in ("sp", "tr"):
                    full_id = "|".join(parts[:3])
                    accession = parts[1]
                else:
                    full_id = header.split()[0]
                    accession = full_id
                current_id = full_id
                seqs[full_id] = ("", current_desc)
                seqs[accession] = ("", current_desc)
                buf = []
            else:
                buf.append(line)
        if current_id is not None:
            seq = "".join(buf)
            seqs[current_id] = (seq, current_desc)
    return seqs


def parse_modification_ini(path):
    mods = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("@") or line.startswith("name"):
                continue
            if "=" not in line:
                continue
            name, rest = line.split("=", 1)
            parts = rest.split()
            if len(parts) < 4:
                continue
            residues = parts[0]
            mass = None
            try:
                mass = float(parts[2])
            except ValueError:
                mass = None
            mods[name] = {"residues": residues, "mass": mass}
    return mods


def parse_plink_params(path):
    params = {
        "fix_mods": [],
        "var_mods": [],
        "enzyme": None,
        "missed_cleavages": None,
        "precursor_tol": None,
        "precursor_tol_unit": None,
        "fragment_tol": None,
        "fragment_tol_unit": None,
        "linker": None,
    }
    section = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if "=" not in line:
                continue
            key, value = [p.strip() for p in line.split("=", 1)]
            if section == "database":
                if key == "enzyme_name":
                    params["enzyme"] = value
                elif key == "max_miss_site":
                    try:
                        params["missed_cleavages"] = int(value)
                    except ValueError:
                        params["missed_cleavages"] = None
            elif section == "modification":
                if key.startswith("fix_mod"):
                    params["fix_mods"].append(value)
                if key.startswith("var_mod"):
                    params["var_mods"].append(value)
            elif section == "ions":
                if key == "peptide_tol":
                    params["precursor_tol"] = value
                elif key == "peptide_tol_type":
                    params["precursor_tol_unit"] = value
                elif key == "fragment_tol":
                    params["fragment_tol"] = value
                elif key == "fragment_tol_type":
                    params["fragment_tol_unit"] = value
            elif section == "linker":
                if key == "linker1":
                    params["linker"] = value
    return params


def parse_proteins_field(proteins_field):
    if not proteins_field:
        return None, None
    first_pair = proteins_field.split("/")[0]
    parts = first_pair.split("-")
    if len(parts) != 2:
        return None, None

    def _extract_id(part):
        part = part.strip()
        if "(" in part:
            part = part.split("(", 1)[0].strip()
        return part

    return _extract_id(parts[0]), _extract_id(parts[1])


def find_peptide_in_protein(protein_seq, peptide):
    if not protein_seq:
        return None
    idx = protein_seq.find(peptide)
    if idx == -1:
        return None
    return idx + 1


def build_mzid(csv_path, fasta_path, mgf_path, plink_path, mod_path, output_path):
    fasta = parse_fasta(fasta_path)
    mod_map = parse_modification_ini(mod_path)
    plink = parse_plink_params(plink_path)

    root = ET.Element(f"{{{NS['mzid']}}}MzIdentML")
    root.set("id", "Deep4D_XL")
    root.set("version", "1.3.0")
    root.set("creationDate", datetime.datetime.now().isoformat(timespec="seconds"))

    cv_list = ET.SubElement(root, f"{{{NS['mzid']}}}cvList")
    for cv_id, uri, name in [
        ("PSI-MS", "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo", "PSI-MS"),
        ("UNIMOD", "http://www.unimod.org/obo/unimod.obo", "UNIMOD"),
        ("XLMOD", "https://raw.githubusercontent.com/HUPO-PSI/xlmod-CV/main/XLMOD.obo", "XLMOD"),
    ]:
        cv = ET.SubElement(cv_list, f"{{{NS['mzid']}}}cv")
        cv.set("id", cv_id)
        cv.set("fullName", name)
        cv.set("uri", uri)

    cvp = ET.SubElement(root, f"{{{NS['mzid']}}}cvParam")
    cvp.set("accession", "MS:1003385")
    cvp.set("name", "mzIdentML crosslinking extension document version")
    cvp.set("cvRef", PSI_MS)
    cvp.set("value", "1.0.0")

    asl = ET.SubElement(root, f"{{{NS['mzid']}}}AnalysisSoftwareList")
    sw = ET.SubElement(asl, f"{{{NS['mzid']}}}AnalysisSoftware")
    sw.set("id", "Deep4D_XL")
    sw.set("name", "Deep4D_XL")
    sw.set("version", "unknown")
    sw_name = ET.SubElement(sw, f"{{{NS['mzid']}}}SoftwareName")
    _cv_param(sw_name, "MS:1001456", "analysis software", PSI_MS)

    audit = ET.SubElement(root, f"{{{NS['mzid']}}}AuditCollection")
    org = ET.SubElement(audit, f"{{{NS['mzid']}}}Organization")
    org.set("id", "ORG_1")
    org.set("name", "Deep4D_XL")
    provider = ET.SubElement(root, f"{{{NS['mzid']}}}Provider")
    contact_role = ET.SubElement(provider, f"{{{NS['mzid']}}}ContactRole")
    contact_role.set("contact_ref", "ORG_1")
    role = ET.SubElement(contact_role, f"{{{NS['mzid']}}}Role")
    _cv_param(role, "MS:1001271", "researcher", PSI_MS)

    seq_collection = ET.SubElement(root, f"{{{NS['mzid']}}}SequenceCollection")
    db_sequences = {}
    peptides = {}
    peptide_evidence = {}

    data_collection = ET.SubElement(root, f"{{{NS['mzid']}}}DataCollection")
    inputs = ET.SubElement(data_collection, f"{{{NS['mzid']}}}Inputs")

    search_db = ET.SubElement(inputs, f"{{{NS['mzid']}}}SearchDatabase")
    search_db.set("id", "SearchDB_1")
    search_db.set("location", fasta_path)
    search_db.set("name", os.path.basename(fasta_path))
    _cv_param(search_db, "MS:1001073", "database type amino acid", PSI_MS)

    spectra_data = ET.SubElement(inputs, f"{{{NS['mzid']}}}SpectraData")
    spectra_data.set("id", "SpectraData_1")
    spectra_data.set("location", mgf_path)
    spectra_data.set("name", os.path.basename(mgf_path))
    spectrum_id_format = ET.SubElement(spectra_data, f"{{{NS['mzid']}}}SpectrumIDFormat")
    _cv_param(spectrum_id_format, "MS:1000774", "multiple peak list nativeID format", PSI_MS)
    file_format = ET.SubElement(spectra_data, f"{{{NS['mzid']}}}FileFormat")
    _cv_param(file_format, "MS:1001062", "Mascot MGF file", PSI_MS)

    apc = ET.SubElement(root, f"{{{NS['mzid']}}}AnalysisProtocolCollection")
    sip = ET.SubElement(apc, f"{{{NS['mzid']}}}SpectrumIdentificationProtocol")
    sip.set("id", "SIP_1")
    sip.set("analysisSoftware_ref", "Deep4D_XL")
    search_type = ET.SubElement(sip, f"{{{NS['mzid']}}}SearchType")
    _cv_param(search_type, "MS:1001083", "ms-ms search", PSI_MS)

    add_params = ET.SubElement(sip, f"{{{NS['mzid']}}}AdditionalSearchParams")
    params_list = ET.SubElement(add_params, f"{{{NS['mzid']}}}ParamList")
    _cv_param(params_list, "MS:1002494", "crosslinking search", PSI_MS)

    if plink.get("enzyme"):
        enzymes = ET.SubElement(sip, f"{{{NS['mzid']}}}Enzymes")
        enz = ET.SubElement(enzymes, f"{{{NS['mzid']}}}Enzyme")
        enz.set("id", "ENZ_1")
        enz.set("missedCleavages", str(plink.get("missed_cleavages") or 0))
        enz_name = ET.SubElement(enz, f"{{{NS['mzid']}}}EnzymeName")
        _cv_param(enz_name, "MS:1001251", plink["enzyme"], PSI_MS)

    frag_tol = ET.SubElement(sip, f"{{{NS['mzid']}}}FragmentTolerance")
    prec_tol = ET.SubElement(sip, f"{{{NS['mzid']}}}ParentTolerance")

    def _tol_to_unit(unit):
        if unit == "ppm":
            return ("UO:0000169", "parts per million", "UO")
        return ("UO:0000221", "dalton", "UO")

    if plink.get("fragment_tol"):
        uacc, uname, uref = _tol_to_unit(plink.get("fragment_tol_unit"))
        _cv_param(frag_tol, "MS:1001412", "search tolerance plus value", PSI_MS, value=plink["fragment_tol"], unit_accession=uacc, unit_name=uname, unit_cv_ref=uref)
        _cv_param(frag_tol, "MS:1001413", "search tolerance minus value", PSI_MS, value=plink["fragment_tol"], unit_accession=uacc, unit_name=uname, unit_cv_ref=uref)

    if plink.get("precursor_tol"):
        uacc, uname, uref = _tol_to_unit(plink.get("precursor_tol_unit"))
        _cv_param(prec_tol, "MS:1001412", "search tolerance plus value", PSI_MS, value=plink["precursor_tol"], unit_accession=uacc, unit_name=uname, unit_cv_ref=uref)
        _cv_param(prec_tol, "MS:1001413", "search tolerance minus value", PSI_MS, value=plink["precursor_tol"], unit_accession=uacc, unit_name=uname, unit_cv_ref=uref)

    mods = ET.SubElement(sip, f"{{{NS['mzid']}}}ModificationParams")

    for mod_name in plink.get("fix_mods", []):
        if not mod_name:
            continue
        mod_entry = mod_map.get(mod_name)
        sm = ET.SubElement(mods, f"{{{NS['mzid']}}}SearchModification")
        sm.set("fixedMod", "true")
        if mod_entry and mod_entry.get("mass") is not None:
            sm.set("massDelta", str(mod_entry["mass"]))
        residues = mod_entry["residues"] if mod_entry else ""
        sm.set("residues", residues if residues else "C")
        if mod_name.lower().startswith("carbamidomethyl"):
            _cv_param(sm, "UNIMOD:4", "carbamidomethyl", UNIMOD)
        else:
            _cv_param(sm, "MS:1001460", mod_name, PSI_MS)

    for is_donor in (True, False):
        sm = ET.SubElement(mods, f"{{{NS['mzid']}}}SearchModification")
        sm.set("fixedMod", "false")
        sm.set("residues", "K")
        sm.set("massDelta", "138.06807961" if is_donor else "0")
        _cv_param(sm, "MS:1003392", "search modification id", PSI_MS, value="crosslink_donor" if is_donor else "crosslink_acceptor")
        _cv_param(sm, "XLMOD:02001", "DSS", XLMOD)
        _cv_param(sm, "MS:1002509" if is_donor else "MS:1002510", "crosslink donor" if is_donor else "crosslink acceptor", PSI_MS, value="0")

    analysis_collection = ET.SubElement(root, f"{{{NS['mzid']}}}AnalysisCollection")
    spec_ident = ET.SubElement(analysis_collection, f"{{{NS['mzid']}}}SpectrumIdentification")
    spec_ident.set("id", "SpecIdent_1")
    spec_ident.set("spectrumIdentificationList_ref", "SIL_1")
    spec_ident.set("spectrumIdentificationProtocol_ref", "SIP_1")

    analysis_data = ET.SubElement(data_collection, f"{{{NS['mzid']}}}AnalysisData")
    sil = ET.SubElement(analysis_data, f"{{{NS['mzid']}}}SpectrumIdentificationList")
    sil.set("id", "SIL_1")
    sil.set("numSequencesSearched", "0")
    sil.set("spectrumIdentificationListRef", "SIL_1")

    group_id = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            pep1 = row.get("peptide1") or ""
            pep2 = row.get("peptide2") or ""
            if not pep1 or not pep2:
                continue
            protein1, protein2 = parse_proteins_field(row.get("Proteins", ""))
            if not protein1 or not protein2:
                continue

            group_id += 1
            spec_id = row.get("Title") or row.get("title") or row.get("TITLE") or f"scan={row_idx + 1}"

            sir = ET.SubElement(sil, f"{{{NS['mzid']}}}SpectrumIdentificationResult")
            sir.set("id", f"SIR_{group_id}")
            sir.set("spectrumID", spec_id)
            sir.set("spectraData_ref", "SpectraData_1")

            for side, pep, protein, site in (
                ("alpha", pep1, protein1, row.get("site1")),
                ("beta", pep2, protein2, row.get("site2")),
            ):
                pep_id = f"PEP_{pep}_{protein}"
                if pep_id not in peptides:
                    pep_el = ET.SubElement(seq_collection, f"{{{NS['mzid']}}}Peptide")
                    pep_el.set("id", pep_id)
                    pep_seq_el = ET.SubElement(pep_el, f"{{{NS['mzid']}}}PeptideSequence")
                    pep_seq_el.text = pep

                    if site:
                        try:
                            site_pos = int(float(site))
                        except ValueError:
                            site_pos = None
                        if site_pos and 1 <= site_pos <= len(pep):
                            mod_el = ET.SubElement(pep_el, f"{{{NS['mzid']}}}Modification")
                            mod_el.set("location", str(site_pos))
                            mod_el.set("residues", pep[site_pos - 1])
                            mod_el.set("monoisotopicMassDelta", "138.06807961" if side == "alpha" else "0")
                            _cv_param(mod_el, "MS:1003393", "search modification id ref", PSI_MS, value="crosslink_donor" if side == "alpha" else "crosslink_acceptor")
                            if side == "alpha":
                                _cv_param(mod_el, "XLMOD:02001", "DSS", XLMOD)
                                _cv_param(mod_el, "MS:1002509", "crosslink donor", PSI_MS, value=str(group_id))
                            else:
                                _cv_param(mod_el, "MS:1002510", "crosslink acceptor", PSI_MS, value=str(group_id))

                    peptides[pep_id] = pep_el

                if protein not in db_sequences:
                    protein_seq, protein_desc = fasta.get(protein, ("", protein))
                    if not protein_seq:
                        accession = protein.split("|")[1] if "|" in protein else protein
                        protein_seq, protein_desc = fasta.get(accession, ("", protein))
                    db_seq_el = ET.SubElement(seq_collection, f"{{{NS['mzid']}}}DBSequence")
                    db_seq_el.set("id", protein)
                    db_seq_el.set("accession", protein)
                    db_seq_el.set("searchDatabase_ref", "SearchDB_1")
                    if protein_desc:
                        _cv_param(db_seq_el, "MS:1001088", "protein description", PSI_MS, value=protein_desc)
                    seq_el = ET.SubElement(db_seq_el, f"{{{NS['mzid']}}}Seq")
                    seq_el.text = protein_seq
                    db_sequences[protein] = protein_seq

                pe_key = f"{pep_id}_{protein}"
                if pe_key not in peptide_evidence:
                    protein_seq = db_sequences.get(protein, "")
                    start = find_peptide_in_protein(protein_seq, pep) if protein_seq else None
                    if start is None:
                        continue
                    end = start + len(pep) - 1
                    pe = ET.SubElement(seq_collection, f"{{{NS['mzid']}}}PeptideEvidence")
                    pe.set("id", pe_key)
                    pe.set("peptide_ref", pep_id)
                    pe.set("dBSequence_ref", protein)
                    pe.set("start", str(start))
                    pe.set("end", str(end))
                    pe.set("isDecoy", "false")
                    pre = protein_seq[start - 2] if start > 1 else "-"
                    post = protein_seq[end] if end < len(protein_seq) else "-"
                    pe.set("pre", pre)
                    pe.set("post", post)
                    peptide_evidence[pe_key] = pe

                sii = ET.SubElement(sir, f"{{{NS['mzid']}}}SpectrumIdentificationItem")
                sii.set("id", f"SII_{group_id}_{side}")
                sii.set("rank", "1")
                sii.set("passThreshold", "true")
                sii.set("peptide_ref", pep_id)
                sii.set("chargeState", str(int(float(row.get("Charge") or row.get("charge") or 0))))
                sii.set("experimentalMassToCharge", str(row.get("m_z") or row.get("m_z")))
                sii.set("calculatedMassToCharge", str(row.get("m_z") or row.get("m_z")))

                pe_ref = ET.SubElement(sii, f"{{{NS['mzid']}}}PeptideEvidenceRef")
                pe_ref.set("peptideEvidence_ref", pe_key)

                _cv_param(sii, "MS:1002511", "crosslink spectrum identification item", PSI_MS, value=f"SII_{group_id}_link")

                ml_score = row.get("ml_score")
                if ml_score:
                    _cv_param(sii, "MS:1001153", "search engine specific score", PSI_MS, value=ml_score)
                    _user_param(sii, "ml_score", ml_score)
                fdr = row.get("FDR")
                if fdr:
                    _cv_param(sii, "MS:1003337", "crosslinked PSM-level global FDR", PSI_MS, value=fdr)

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    pretty_lines = [line for line in pretty.splitlines() if line.strip()]
    with open(output_path, "wb") as fh:
        fh.write(b"\n".join(pretty_lines) + b"\n")
