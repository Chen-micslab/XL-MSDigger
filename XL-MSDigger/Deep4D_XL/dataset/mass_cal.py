import numpy as np
import pandas as pd
import math
from Deep4D_XL.dataset.constant import Mass
import time


class mz_cal():                       
    def calculate_ccs(self, peptide_m_z, peptide_charge, peptide_k0):
        m = 28.00615
        t = 304.7527
        coeff = 18500 * peptide_charge * math.sqrt(
            (peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
        ccs = coeff * peptide_k0
        return ccs

    def calculate_k0(self, peptide_m_z, peptide_charge, peptide_ccs):
        m = 28.00615
        t = 304.7527
        coeff = 18500 * peptide_charge * math.sqrt(
            (peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
        k0 = peptide_ccs / coeff
        return k0

    def extract_from_cl_peptide(self, cl_peptide):                                
        id1 = cl_peptide.find('X')
        pep1 = cl_peptide[:id1]
        pep2 = cl_peptide[(id1 + 1):]
        site1 = pep1.find('U')
        site2 = pep2.find('U')
        peptide1 = pep1.replace('U', '')
        peptide2 = pep2.replace('U', '')
        return peptide1, peptide2, site1, site2

    def regular_peptide_m_z(self, peptide, charge):
        mass = 0
        for i in range(len(peptide)):
            mass = mass + Mass.AA_residue_mass[peptide[i]]
        mass = mass + Mass.loss_mass['H2O']
        m_z = mass / charge + Mass.loss_mass['H+']
        return m_z

    def crosslink_peptide_m_z(self, cl_peptide, charge, crosslinker):
        peptide1, peptide2, site1, site2 = self.extract_from_cl_peptide(cl_peptide)
        mass1 = self.regular_peptide_m_z(peptide1, 1) - Mass.loss_mass['H+']
        mass2 = self.regular_peptide_m_z(peptide2, 1) - Mass.loss_mass['H+']
        mass = Mass.crosslinker_mass[crosslinker] + mass1 + mass2
        m_z = mass / charge + Mass.loss_mass['H+']
        return m_z

    def regular_peptide_msms_m_z(self, peptide, by_type, num, charge, loss):                         
        if charge > 2:                      
            return -1
        else:
            if by_type == 'b':
                b = peptide
            elif by_type == 'y':
                b = peptide[::-1]
            mass = 0
            for j in range(num):
                mass = mass + Mass.AA_residue_mass[b[j]]
            mass = mass - Mass.loss_mass[loss]
            if by_type == 'b':
                mass = mass + 1.00783         
                m_z = mass / charge + (charge - 1)*1.00728/charge                                        
            elif by_type == 'y':
                mass = mass + 17.00274         
                m_z = mass / charge + (charge + 1)*1.00728/charge                                                  
            if (m_z < 200) or (m_z > 1700):
                return -1
            else:
                return m_z

    def peptide_mass(self, peptide):
        return sum(Mass.AA_residue_mass[aa] for aa in peptide) + Mass.loss_mass['H2O']

    def calculate_m_z(self, mass, charge, charge_offset):
        return mass / charge + (charge + charge_offset) * 1.00728 / charge

    def crosslink_peptide_msms_m_z(self, cl_peptide, crosslinker, by_type, num, charge, loss):
        peptide1, peptide2, site1, site2 = self.extract_from_cl_peptide(cl_peptide)
        peptide1_mass = self.peptide_mass(peptide1)
        peptide2_mass = self.peptide_mass(peptide2)
        mass = 0
        m = 1
        if by_type in ('1b', '1y', '2b', '2y'):
            a = (peptide1 if by_type[0] == '1' else peptide2)[::-1 if by_type[1] == 'y' else 1]

            if num >= len(a):
                return -1
            mass = sum(Mass.AA_residue_mass[a[i]] for i in range(num))

            if by_type[1] == 'b':
                mass += 1.00783         
                charge_offset = -1
            else:
                mass += 17.00274        
                charge_offset = 1

            site = site1 if by_type[0] == '1' else site2
            lenth = len(peptide1) if by_type[0] == '1' else len(peptide2)
            site = site if by_type[1] == 'b' else (lenth - site + 1)
            other_peptide_mass = peptide2_mass if by_type[0] == '1' else peptide1_mass

            if num >= site:
                if charge not in range(2, 6):
                    return -1
                mass += other_peptide_mass + Mass.crosslinker_mass[crosslinker]
            elif charge > 2:
                return -1

            mass -= Mass.loss_mass[loss]
            m_z = self.calculate_m_z(mass, charge, charge_offset)
        else:
            return -1
        return m_z
                                
                        
               
                       

    def mono_peptide_msms_m_z(self, peptide, site, crosslinker, by_type, num, charge, loss):
        if by_type == 'b':
            b = peptide
        elif by_type == 'y':
            b = peptide[::-1]
        mass = 0
        for j in range(num):
            mass = mass + Mass.AA_residue_mass[b[j]]
        mass = mass - Mass.loss_mass[loss]
        if by_type == 'b':
            if num >= site:
                mass = mass + Mass.crosslinker_hydro_mass[crosslinker]
            mass = mass + 1.00783         
            m_z = mass / charge + (charge - 1)*1.00728/charge                                        
        elif by_type == 'y':
            if num >= (len(peptide) - site + 1):
                mass = mass + Mass.crosslinker_hydro_mass[crosslinker]
            mass = mass + 17.00274         
            m_z = mass / charge + (charge + 1)*1.00728/charge                                                  
        return m_z

    def loop_peptide_msms_m_z(self, peptide, site1, site2, crosslinker, by_type, num, charge, loss):                  
        if by_type == 'b':
            b = peptide
        elif by_type == 'y':
            b = peptide[::-1]
        mass = 0
        for j in range(num):
            mass = mass + Mass.AA_residue_mass[b[j]]
        mass = mass - Mass.loss_mass[loss]
        if by_type == 'b':
            if num >= site2:
                mass = mass + Mass.crosslinker_mass[crosslinker]
            mass = mass + 1.00783         
            m_z = mass / charge + (charge - 1)*1.00728/charge                                        
        elif by_type == 'y':
            if num >= (len(peptide) - site1 + 1):
                mass = mass + Mass.crosslinker_mass[crosslinker]
            mass = mass + 17.00274         
            m_z = mass / charge + (charge + 1)*1.00728/charge                                                  
        if num >= site1 and num < site2:
            m_z = 0
        return m_z

if __name__ == '__main__':
    a = mz_cal()
    aa = time.perf_counter()
    print(a.crosslink_peptide_msms_m_z1('QAMIRAMETLKUILYKXPESRLCLKUPSDLR', 'DSS', '2y', 7, 2, 'noloss'))
    bb = time.perf_counter()
    print(bb - aa)
    aa = time.perf_counter()
    print(a.crosslink_peptide_msms_m_z('QAMIRAMETLKUILYKXPESRLCLKUPSDLR', 'DSS','2y', 7, 2,'noloss'))
    bb = time.perf_counter()
    print(bb - aa)
                                                                                    
                                                                 
              

