# Updated script with the flag "in_main_data"
import re

def parse_string(s):
    conic = 0
    parms = [0]*10  # Initialize list with zeros

    k_search = re.search(r"K\s*=\s*([-−\d\.eE+]+)", s)
    if k_search:
        conic = float(k_search.group(1))

    #a_search = re.findall(r"A\s*(\d)\s*=\s*([-\d\.e+]+)", s)
    a_search = re.findall(r"A\s*(\d+)\s*=\s*([-−\d\.eE+]+)", s)
    print('a_search',a_search)
    for a in a_search:
        index = int(a[0])//2 - 1  # Calculate the index for the array
        if index > len(parms)-1:
            print('need longer parms')
#        print(a,a_search)
        parms[index] = float(a[1].replace('−', '-'))

    return conic, parms

def distance_pattern(s):
    # If the string matches the pattern 'variable' or '(variable)'
    if re.fullmatch(r'\(?\bvariable\b\)?', s):
        return True
    # If the string matches the pattern 'dx' or '(dx)', where x is an integer
    elif re.fullmatch(r'\(?\bd\d+\b\)?', s):
        return True
    else:
        return False
def parse_optical_file(file_path,loc=0):
    # Start by reading the file
    withgf=0
    withdgf=0
    withdiam=0
    withrad=0
    withglass=0
    offset=-1
    with open(file_path, "r") as file:
        content = file.readlines()

    # Initialize the dictionary
    optical_data = {}

    # Initialize variable to keep track of the aspherical data section
    in_aspherical_section = False

    # List to temporarily store the aspherical parameters
    aspherical_parms = []

    # Variable to temporarily store the aspherical surface number
    aspherical_surface_no = None

    # Initialize a dictionary to store variable surface locations
    variable_surface_locations = {}

    # List to store variable surfaces for processing later
    variable_surfaces = []

    # Variable to track if we are in the main data section
    in_main_data = False

    # Start reading from the 5th line where the actual data starts
    for line in content[:]:
        #parts = line.split('\t')
        parts = line.split()
        
        # Skip lines that don't contain data (these might be empty lines or headers)
        if len(parts) < 2:
            parts=line.split()
            if len(parts) <2:
                continue

        # Check if we're in the main data section
        if in_main_data:
            if not parts[0].strip().isdigit():
                print('asteric',parts[0].strip(),parts[0].strip().endswith('*'))
                if not parts[0].strip().endswith('*'):
                    print('end of main',parts)
                    in_main_data = False
            # Let's add try-except blocks to handle potential errors
            try:
                print('main',parts)
                surface_no = int(parts[0].replace('*', ''))  # Keep as int, remove any '*' marking aspherical surfaces
                r = float(parts[idx_r].replace('−', '-').replace('∞', 'inf'))  # Replace Unicode minus sign and infinity sign

                # For 'd', check if it's 'variable' and if so, set to None and store the surface number for later processing
                #if parts[idx_d].strip().lower() == '(variable)':
                if distance_pattern(parts[idx_d].strip().lower()):
                    d = None
                    variable_surfaces.append(surface_no)
                elif parts[idx_d].strip().lower() == 'bf' or parts[idx_d].strip().lower() == '(bf)':
                    d = None
                    variable_surfaces.append(surface_no)
                else:
                    d = float(parts[idx_d].replace('−', '-'))  # Replace Unicode minus sign

                # For nd and vd, fill in with 1 and None respectively if they're not provided
                nd = float(parts[idx_nd].replace('−', '-')) if len(parts) > 3 else 1.0
                vd = float(parts[idx_vd].replace('−', '-')) if len(parts) > 4 else None
                if withdiam:
                    print('withdiam',idx_diam)
                    diam = float(parts[idx_diam].replace('−', '-'))
                if withrad:
                    diam = 2*float(parts[idx_rad].replace('−', '-')) if len(parts) > 5+withgf+withdgf else 5.0
                #if withgf:
                #    diam = float(parts[6].replace('−', '-')) if len(parts) > 6 else 5.0
                if withglass:
                    glass = parts[idx_glass].replace('−', '-')
                
                # Set the type of the surface
                if '*' in parts[0]:
                    surface_type = 'EVENASPH'
                    conic = 0.0
                    aspherical_parms = [0.0]
                else:
                    surface_type = 'STANDARD'
                    conic = 0.0
                    aspherical_parms = [0.0]
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Error: {e}")
                print(line.split())
                print(line.split('\t'))
                continue

            # Store in the dictionary
            print('stored surface No.:',surface_no)
            optical_data[surface_no] = {'r': r, 'd': d, 'nd': nd, 'vd': vd, 'type': surface_type, 'conic': conic, 'parm': aspherical_parms, 'diam': diam, 'glass': glass}
            print(optical_data[surface_no]) 
            continue

        # Check if this line starts the main data section
        if parts[0].strip().lower() == 'surface' and ( parts[1].strip().lower() == 'number' or parts[1].strip().lower() == 'no.'):
            in_main_data = True
            print('start of main',parts)
            if 'θgF' in line:
                withgf=1
                idx_gf = parts.index('θgF') + offset
            if 'ΔθgF' in line:
                withdgf=1
                idx_dgf = parts.index('ΔθgF') + offset
            if 'diameter' in line:
                withdiam=1
                idx_diam = parts.index('diameter') + offset
            if 'rad' in line:
                withrad=1
                idx_rad = parts.index('rad') + offset
            if 'glass' in line or 'Glass' in line:
                withglass=1
                idx_glass = parts.index('glass') + offset
            else:
                glass = None
            idx_r = parts.index('r') + offset
            idx_d = parts.index('d') + offset
            idx_nd = parts.index('nd') + offset
            if 'vd' in line:
                idx_vd = parts.index('vd') + offset
            else:
                idx_vd = parts.index('vd') + offset


            continue

        # Check if we're in the aspherical data section
        if in_aspherical_section:
            print(f"Aspherical section line: {line}")
            # If this line does not start with 'K', 'A+number', or 'number+"th"/st/nd', it marks the end of the aspherical data section
            parts = line.split()  # Split by spaces instead of tabs
            if aspherical_surface_no:
                print('aspherical_surface_no',aspherical_surface_no)
                optical_data[aspherical_surface_no]['conic'] = conic
                optical_data[aspherical_surface_no]['parm'] = aspherical_parms
                print('readin surf no.',aspherical_surface_no,conic,aspherical_parms)
                #aspherical_parms = []
            if not (parts[0].strip().startswith('K') or parts[0].strip().startswith('A') or parts[0].strip()[0].isdigit()):
                print(f"not Aspherical section line: {line}")
                in_aspherical_section = False
                # Before we end the aspherical data section, update the previous aspherical surface data
                if aspherical_surface_no:
                    optical_data[aspherical_surface_no]['conic'] = conic
                    optical_data[aspherical_surface_no]['parm'] = aspherical_parms
                    print('readin surf no.',aspherical_surface_no,aspherical_parms)
                    aspherical_parms = []
                continue
            # Check if this line contains the surface number
            if "surface" in line.lower():
                aspherical_surface_no = int(parts[0].strip()[:-2])
                continue
            # Otherwise, this line should contain the conic constant or one of the aspherical parameters
            else:
                # If the line starts with 'K', it contains the conic constant and potentially some of the aspherical parameters
                #print('not start with k?',parts[0].strip())
                if parts[0].strip().startswith('K'):
                    print('conic parts',parts)
                    #print('conic line',line)
                    #conic = float(parts[2].replace('−', '-'))
                    #aspherical_parms = [float(p.split('=')[1].replace('−', '-')) for p in parts[2:] if p.strip().startswith('A')]
                    line.replace('−', '-')
                    (conic,aspherical_parms)=parse_string(line)
                    print('parse_string',conic,aspherical_parms)
                # If the line starts with 'A', it contains one or more aspherical parameters
                elif parts[0].strip().startswith('A'):
                    aspherical_parms += [float(p.split('=')[1].replace('−', '-')) for p in parts if p.strip().startswith('A')]
                continue

        # Check if this line starts the aspherical data section
        if "ASPHERIC DATA" in line.upper() or "ASPHERICAL DATA" in line.upper():
            in_aspherical_section = True
            print(f"Starting Aspherical section: {line}")
            continue

        # Check if this line contains variable surface locations
        if parts[0].lower().startswith("d"):
            print(' d for vari',parts)
            variable_surface_no = int(parts[0][1:])  # Extract the surface number
            variable_surface_locations[variable_surface_no] = [float(p.replace('−', '-')) for p in parts[1:] if p.strip()]  # Extract the location data
            continue
        elif parts[0].lower() == 'bf':
            print('bf',parts)
            if len(variable_surfaces) > 0:
                variable_surface_no = max(variable_surfaces)
                variable_surface_locations[variable_surface_no] = [float(p.replace('−', '-')) for p in parts[1:] if p.strip()]
    # Process variable surfaces
    for surface_no in variable_surfaces:
        print('variable', variable_surface_locations,surface_no)
        optical_data[surface_no]['d'] = variable_surface_locations.get(surface_no, [None])[loc]  # Use the first location value if available
    print('finished reading txt')
    return optical_data

# Test
#optical_data = parse_optical_file("/mnt/data/rf8514_1air.txt")
#optical_data

