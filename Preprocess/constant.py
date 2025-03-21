class Mass:
    AA_residue_mass = { 'A': 71.0371138,
               'c': 103.00918,
               'C': 160.0306481,
               'D': 115.0269429,
               'E': 129.042593,
               'F': 147.0684139,
               'G': 57.0214637,
               'H': 137.0589118,
               'I': 113.0840639,
               'K': 128.094963,
               'L': 113.0840639,
               'M': 131.0404846,
               'N': 114.0429274,
               'P': 97.0527638,
               'Q': 128.0585774,
               'R': 156.101111,
               'S': 87.0320284,
               'T': 101.0476784,
               'V': 99.0684139,
               'W': 186.0793129,
               'Y': 163.0633285,
               'a': 42.01056,
               's': 166.9983594,
               't': 181.0140094,
               'y': 243.0296595,
               'e': 147.0353996,
               'B': 147.0353996}

    loss_mass = {'noloss': 0,
                 'Noloss': 0,
                 'NH3': 17.026548,
                 'H2O': 18.010565,
                 'H3PO4': 97.977,
                 'H+': 1.00728,
                 'H': 1.00783
                 }

    crosslinker_mass = {'DSS': 138.0680795652,
                        'BS3': 138.0680795652,
                        'DSSO': 158.0037647489,
                        'DSBU': 196.0847922619,
                        'SO2F_424': 389.116,
                        'SO2F_467': 475.153,
                        'NHSF_SO3': 289.041,
                        'SO2F467': 332.047}

    crosslinker_hydro_mass = {'DSS': 138.0680795652 + loss_mass['H2O'],
                              'BS3': 138.0680795652 + loss_mass['H2O'],
                              'DSSO': 158.0037647489 + loss_mass['H2O'],
                              'DSBU': 196.0847922619 + loss_mass['H2O'],
                              'NHSF_SO3': 289.041 + loss_mass['H2O'],
                              'SO2F467': 332.047 + loss_mass['H2O'],
                              'NHSF_SO3_1': 289.041 + loss_mass['H2O'] + 97.0174,
                              'SO2F467_1': 332.047 + loss_mass['H2O'] + 97.0174}