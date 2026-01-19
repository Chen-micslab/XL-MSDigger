                              
import numpy as np
import pandas as pd
import math
from Preprocess.constant import Mass

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

    def extract_from_mono_peptide(self, mono_peptide):                                
        site1 = mono_peptide.find('U')
        peptide1 = mono_peptide.replace('U', '')
        return peptide1, site1

    def extract_from_loop_peptide(self, loop_peptide):                                
        site1 = loop_peptide.find('U')
        site2 = loop_peptide.rfind('U')-1
        peptide1 = loop_peptide.replace('U', '')
        return peptide1, site1, site2

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

    def crosslink_peptide_msms_m_z(self, cl_peptide, crosslinker, by_type, num, charge, loss):
        peptide1, peptide2, site1, site2 = self.extract_from_cl_peptide(cl_peptide)
        peptide1_mass = 0
        peptide2_mass = 0
        mass = 0
        m = 1
        for i in range(len(peptide1)):
            peptide1_mass = peptide1_mass + Mass.AA_residue_mass[peptide1[i]]
        for i in range(len(peptide2)):
            peptide2_mass = peptide2_mass + Mass.AA_residue_mass[peptide2[i]]
        peptide1_mass = peptide1_mass + Mass.loss_mass['H2O']
        peptide2_mass = peptide2_mass + Mass.loss_mass['H2O']
        if by_type == '1b':
            a = peptide1
            if num >= len(a):
                m = -1
            else:
                for i in range(num):
                    mass = mass + Mass.AA_residue_mass[a[i]]
                if num >= site1:
                    mass = mass + peptide2_mass + Mass.crosslinker_mass[crosslinker]
                mass = mass + 1.00783          
                mass = mass - Mass.loss_mass[loss]
                m_z = mass / charge + (charge - 1) * 1.00728 / charge                                        
        elif by_type == '1y':
            a = peptide1[::-1]
            if num >= len(a):
                m = -1
            else:
                for i in range(num):
                    mass = mass + Mass.AA_residue_mass[a[i]]
                if num >= (len(peptide1) - site1 + 1):
                    mass = mass + peptide2_mass + Mass.crosslinker_mass[crosslinker]
                mass = mass + 17.00274         
                mass = mass - Mass.loss_mass[loss]
                m_z = mass / charge + (charge + 1) * 1.00728 / charge                                                 
        elif by_type == '2b':
            a = peptide2
            if num >= len(a):
                m = -1
            else:
                for i in range(num):
                    mass = mass + Mass.AA_residue_mass[a[i]]
                if num >= site2:
                    mass = mass + peptide1_mass + Mass.crosslinker_mass[crosslinker]
                mass = mass + 1.00783          
                mass = mass - Mass.loss_mass[loss]
                m_z = mass / charge + (charge - 1) * 1.00728 / charge                                        
        elif by_type == '2y':
            a = peptide2[::-1]
            if num >= len(a):
                m = -1
            else:
                for i in range(num):
                    mass = mass + Mass.AA_residue_mass[a[i]]
                if num >= (len(peptide2) - site2 + 1):
                    mass = mass + peptide1_mass + Mass.crosslinker_mass[crosslinker]
                mass = mass + 17.00274         
                mass = mass - Mass.loss_mass[loss]
                m_z = mass / charge + (charge + 1) * 1.00728 / charge                                                 
        if m == -1:
            m_z = -1
        if (m_z < 200) or (m_z > 1700):
            m_z = -1
        return m_z


    def peptide_mass(self, peptide):
        return sum(Mass.AA_residue_mass[aa] for aa in peptide) + Mass.loss_mass['H2O']

    def calculate_m_z(self, mass, charge, charge_offset):
        return mass / charge + (charge + charge_offset) * 1.00728 / charge

    def mono_peptide_msms_m_z(self, mono_peptide, crosslinker, by_type, num, charge, loss):
        peptide, site = self.extract_from_mono_peptide(mono_peptide)
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

    def loop_peptide_msms_m_z(self, loop_peptide, crosslinker, by_type, num, charge, loss):                  
        peptide, site1, site2 = self.extract_from_loop_peptide(loop_peptide)
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
            if num >= site1 and num < site2:
                m_z = -1
        elif by_type == 'y':
            if num >= (len(peptide) - site1 + 1):
                mass = mass + Mass.crosslinker_mass[crosslinker]
            mass = mass + 17.00274         
            m_z = mass / charge + (charge + 1)*1.00728/charge                                                  
            if num >= (len(peptide) - site2 + 1) and num < (len(peptide) - site1 + 1):
                m_z = -1
        return m_z


if __name__ == '__main__':
    a = mz_cal()
                 
                             
                                                                                                      
                             
                
                             
                                                                                                       
                             
                
                             
                                                                             
                             
                
                                          
                 
    print(a.crosslink_peptide_m_z('EKELSKUKXEQKUELK',3,'DSS'))
                                                                 
                               

