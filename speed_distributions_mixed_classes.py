# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:49:05 2026

@author: neyda
"""
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import curve_fit
import math
import pandas as pd
from scipy.signal import savgol_filter
import os 
#PARAMETERS OF THE SYSTEM

#Then we can change the code to work to link this parameters with the ones of each video

#PARAMETERS
#Change depending on the number of classes extra we want to add in the system   
particle_classes_id = np.array([1, 2, 3])
particle_type_class = np.array(["Metallic","Small_Plastics","Big_Plastics"])

class_radius = {
    1: 0.9/2,   # metallics
    2: 1.0/2,  # small plastics
    3: 1.0/2   # big plastics
}
    
THRES = 0.1
num_bins = 10
window_length = 101
polyorder = 2
n_same_vid = 5 #number of repetitions of every experiment
mean_val = []#list of values for ALL THE DAYS experiments 
N_met_list = []
met_pf = []#save data graphics Nmet/Ntotal vs vx_mean values of diff part. types
vx_mean_den = {i: [] for i in range(len(particle_classes_id) + 1)}#save data graphics Nmet/Ntotal vs vx_mean values of diff part. types
#saving lists to save all dfs to do the mean of every 5 repeated videos
dfs_block = []
sp_block = []
sp_block_segregated = [[]for _ in range(len(particle_classes_id)+1)]#index 1,2 and 3
vx_block = []
vx_block_segregated = [[]for _ in range(len(particle_classes_id)+1)]
vy_block = []
vy_block_segregated = [[]for _ in range(len(particle_classes_id)+1)]


dfs_block_segregated = [[] for _ in range(len(particle_classes_id)+1)] #the same but separated regarding the particles classes
max_time = 50 #time considered for doing the graphics of all videos
max_frames = max_time*30 #30 FPS, all the videos last at least 50 seconds, time range for the mean values of repeated videos
day = 17
month = 7
N_videos =26
distance = 15
Area_box = 15*distance # 15 is the width


#AUXILIARY FONCTIONS
class_colors = {
    1: "blue",   # metallics
    2: "red",    # small plastics
    3: "yellow"   # big plastics
}

#calculates the max desviation of the distribution of velocities for une particule. That is going 
def threshold_values(vect):
    if len(vect) < 2:#if there is less than two values, statistics does not work
        return 0
    else: 
        desv = statistics.stdev(vect)
        threshold_value = desv*0.3
        return threshold_value
    
#to be our threshold, variable for each particle
def threshold(vect):
    if len(vect) < 2:#if there is less than two values, statistics does not work
        return 0.0
    else: 
        desv = statistics.stdev(vect)
        return 0.2*desv
    
#get the area ocuppied by the different types of particles
def get_class_area(class_id, class_radius):
    if class_id == 1:
        area = class_radius[1]*class_radius[1]*math.pi
    elif class_id == 2:
        area = class_radius[2]*class_radius[2]*math.pi
    elif class_id == 3:
        area = class_radius[3]*class_radius[3]*math.pi
    else:
        print("Not identified class")
    return area

#Eliminates de velocities under the threshold for une particle. Returns one array with the values 
#filtered
def eliminate_vel (vect, thres):
    velocities = []
    if len(vect) == 0 or vect is None:
        print("Empty vector, not possible to eliminate velocities")
        return np.zeros(len(vect)+1)
    for i in range(len(vect)):
        if vect[i] is None:
            print("Imput vector has a None component")
            continue
        if abs(vect[i]) > abs(thres):
            velocities.append(vect[i])    
    return np.array(velocities)

#Obtains the temporal average of the speed for one particle
def temporal_prom(vect):
    frames = len(vect)
    vel = 0
    for i in range(len(vect)):
        vel += vect[i]
    if frames > 0:
        vel = vel/frames 
        return vel
    else :
        return 0

#classify one particle acording the number associated to every class
def get_class_name(class_id):
    if class_id == 1:
        p_class = "Metallic"
    elif class_id == 2:
        p_class = "Small Plastic"
    elif class_id == 3:
        p_class = "Big Plastic"
    else:
        print("Not identified class")
    return p_class

# Asociates a mass value for every particle class
def get_class_weight(class_id):
    if class_id == 1:
        class_w = 1.297
    elif class_id == 2:
        class_w = 0.496
    elif class_id == 3:
        class_w = 1.0 #no pesan esto, pero me lo invento
    else:
        print("Not identified class")
    return class_w 

#check if the number of particles detected is the same as the one it is supposed to be
def control_particles(count_classes,particle_type_num,eliminated):
    rest = np.array(count_classes)
    zero = np.zeros(count_classes)
    for i in range(len(count_classes)):
        rest[i] = (count_classes[i] + eliminated[i]) - particle_type_num[i]
        if not np.array_equal(count_classes,particle_type_num) and np.array_equal(rest,zero ) :
            print ("Error in number of particles")
            print(f"particles expected = {particle_type_num}\n")
            print(f"particles in reallity  = {count_classes}")
    return 

#Define the Gaussian function
def Gaussian_func(x,a,mu,sigma,c):#a amplitud, c vertical deplacement
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

#Do the Gaussian_fit  
def Gaussian_fit(vectx, vecty):
    a_init = max(vecty) - min(vecty) if max(vecty) > min(vecty) else max(vecty)#just in case it is not a flat distribution
    mu_init = vectx[np.argmax(vecty)]       
    sigma_init = (max(vectx) - min(vectx)) / 4.0 if (max(vectx) - min(vectx)) > 0 else 1.0#total guassian range is aprox 4 sigmas
    c_init = min(vecty)#vertical desviation
    p0 = [a_init, mu_init, sigma_init, c_init]
    # physical bounds: [a > 0, mu libre, sigma > 0, c >= 0]
    bounds_lower = [0, -np.inf, 1e-4, 0]
    bounds_upper = [np.inf, np.inf, np.inf, np.inf]
    try:
        popt, pcov = curve_fit(
            Gaussian_func, 
            vectx, 
            vecty, 
            p0=p0, 
            bounds=(bounds_lower, bounds_upper), 
            max_nfev=100000
            )
    except RuntimeError:#not possible to find optimal values in max_nfev essays
   # If the histogram does not converge, we use the first approximative values
        print(f"Gaussian did not converged using p0 = {p0} .")
        popt = p0
        pcov = np.zeros((4, 4))
    #print(f"optimal Gaussian values = {popt}")
    perr = np.sqrt(np.diag(pcov))
    #print(f"Error = {perr}")
    return popt, pcov, perr

#get x value and y values for the Gaussian fit
def get_histogram_data(data, num_bins=10):
    counts, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 #x values
    return bin_centers, counts

# Savitzky-Golay filter for 1 array
def apply_savgol_filter(data,window_length,polyorder):
    n = len(data)
    if n <= polyorder or n == 0:
        print(f"vector size = {n}, impossible to apply the filter")
        return data
    if window_length > n:
        window_length = n - 1 if n % 2 == 0 else n
    # window must be impair
    if window_length % 2 == 0:
        window_length -= 1 
    #vheck if wl is smaller than the order
    if window_length <= polyorder:
        print(f"vector size = {n}, window size too small for polyorder")
        return data
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

    #MAIN
  
#creating paths for extracting the data/saving the videos
output_dir_cm = f"C:/Users/Neyda/Documents/Materia_Activa_UNAV/Outputs/d_{distance}/2026_0{month}_{day}/Cm"
output_dir_dist = f"C:/Users/Neyda/Documents/Materia_Activa_UNAV/Outputs/d_{distance}/2026_0{month}_{day}/Speed_Distributions"
imput_dir = f"C:/Users/Neyda/Documents/Materia_Activa_UNAV/Data_videos/d_{distance}/copy_{day}"
os.makedirs(output_dir_cm, exist_ok=True)
os.makedirs(output_dir_dist, exist_ok=True)

#CHECK CONFIGURATION EXCEL IS CORRECT
#read particles number for each video from a excel file:
df_config = pd.read_excel(f"{imput_dir}/Config_Videos.xlsx", dtype={'video_id': str})
df_config['video_id'] = df_config['video_id'].astype(str).str.zfill(4)#tranfor excel column 1 into 0001, adds zeros to the str until having 4 digits, astype(str) change the numbers to str
df_config.set_index('video_id', inplace=True)#this column will be the index of the dataframe
for h in range(1, N_videos +1):
    video_id = f"{h:04d}" #"0001" etc
    filename = f"{imput_dir}/{video_id}_tracking_byParticle.txt"
    if video_id in df_config.index:
        row = df_config.loc[video_id]#extracs videos ids values
        N_Metallics = int(row['N_Metallics'])
        #print(f"N_Metallics = {N_Metallics} for_video ={video_id}")
        N_Small_Plastics = int(row['N_Small_Plastics'])
        #print(f"N_Small_Plastics = {N_Small_Plastics} for_video ={video_id}")
        N_Big_Plastics = int(row['N_Big_Plastics'])
        #print(f"N_Big_Plastics = {N_Big_Plastics} for_video ={video_id}")
    else:
        continue
    particle_type_num = np.array([N_Metallics,N_Small_Plastics,N_Big_Plastics])
    Particle_Num = N_Big_Plastics + N_Small_Plastics + N_Metallics
        #if the video with h index does not exist, we just continue
    if not os.path.exists(filename):
        print(f"Video {video_id} does not exists. Going to the next one...")
        print(f" Proccesing vídeo ID: {video_id} ")
        continue
                   
    frame_id= []#saves the total number of frames in the video
    actual_block = [] # Save temporally the actual bloc values
    all_particles_speeds = []  # Save a list of every bloc
    all_particles_vx = [] #each component is a particle
    all_particles_vy = []
    actual_speeds = [] #for the reading particle
    actual_vx = []
    actual_vy = []
    particles_id = []
    particles_class = []
    speed_particles = []
    speed_particles_segregated = [[] for _ in range(len(particle_type_num)+1)]
    vx_particles_segregated = [[] for _ in range(len(particle_type_num)+1)]
    vy_particles_segregated = [[] for _ in range(len(particle_type_num)+1)]
    vx_particles = []
    vy_particles = []
    registers = []
    clean_registers = []

    #open the file, read the values and extract the paramenters
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()#clean line
            if line.startswith("#"):#if line starts with #, we just ingnore it
                continue

            if not line:#if line end with a space, the bloc ended
                if actual_speeds:  # See if actual_bloc is empty or not. If it is not empty we save values:
                    all_particles_speeds.append(actual_speeds)#each component of the list is a block
                    all_particles_vx.append(actual_vx)  # Guardamos lista de Vx
                    all_particles_vy.append(actual_vy)
                    actual_speeds = []  # Reiniciate for the next bloc
                    actual_vx = [] 
                    actual_vy = [] 
                    continue
        
            data = line.split()#separte the line into columns
            if len(data) >= 8:#read the files. If there is an empty line, we have the if not line code
                p_id = int(data[0])
                p_class = int(data[1])
                # Ignore class_ids differents than 1, 2 o 3
                if p_class not in particle_classes_id:
                    continue
                speed = float(data[7])
                velx = float(data[5])
                vely = float(data[6])
                frame = float(data[2])
                if not actual_speeds:#if it is empty, we save id and class 1 time for particle
                    particles_id.append(p_id)
                    particles_class.append(p_class)
                actual_speeds.append(speed)#acum speeds for one particle
                actual_vx.append(velx)
                actual_vy.append(vely)
                frame_id.append(frame)
                registers.append({#write saved values in the df
                    'frame_id': frame,
                    'particle_id': p_id,
                    'class_id': p_class,
                    'vx': velx,
                    'vy': vely,
                    'speed': speed
                    })

        if actual_speeds:#in case the file does not end with a blanc space
            all_particles_speeds.append(actual_speeds)#save all the last block
            all_particles_vx.append(actual_vx)
            all_particles_vy.append(actual_vy) #Not having the last particle trapped in the actuals lists

    df = pd.DataFrame(registers)
    df["particle_id"] = df["particle_id"].astype(str).str.strip().astype(int)
    p_means = df.groupby("particle_id")[["vx", "vy", "speed"]].mean()#average values for 1p in time for defining the conditions
    df = df.merge(p_means, on="particle_id", suffixes=("", "_mean_val"))#include p_means in df, another df
    #sufix mean_val for the p_mean column. "" indicates the rest columns to rest with the same name, the means values are aligned with the particle_id

    clean_df = df
    #we want only the initial columns
    clean_df = clean_df[["frame_id", "particle_id", "class_id", "vx", "vy", "speed"]]

    #Optimized cm value calculation
    # mass column
    clean_df["mass"] = clean_df["class_id"].apply(get_class_weight)

    #  p = m * v
    clean_df["p_x"] = clean_df["mass"] * clean_df["vx"]
    clean_df["p_y"] = clean_df["mass"] * clean_df["vy"]
    clean_df["p_sp"] = clean_df["mass"] * clean_df["speed"]

    #grouping mean value for every frame
    N_frames = len(clean_df["frame_id"])#get total number of frames
    grouped = clean_df.groupby("frame_id")[["mass", "p_x", "p_y", "p_sp"]].sum().reset_index()#sum all particles values for 1 frame
    #mean value of all frames
    grouped["video_id"] = video_id
    grouped["Nmet"] = N_Metallics
    grouped["speed_mean"] = clean_df.groupby("frame_id")["speed"].mean()#.reindex(grouped.index)
    grouped["vx_mean"] = clean_df.groupby("frame_id")["vx"].mean()
    grouped["vy_mean"] = clean_df.groupby("frame_id")["vy"].mean()
    grouped = grouped.dropna()
    grouped_class = clean_df.groupby((["frame_id", "class_id"]))[["mass", "p_x", "p_y", "p_sp"]].sum().reset_index()
    grouped_class["video_id"] = video_id
    #dfs_block_segregated.append(grouped_class[grouped_class['frame_id'] < max_frames]), Not for the moment
    # ponderated averages
    grouped["p_x"] = (grouped["p_x"] / grouped["mass"]).to_numpy() 
    grouped["p_y"] = (grouped["p_y"] / grouped["mass"]).to_numpy()
    grouped["p_sp"] = (grouped["p_sp"] / grouped["mass"]).to_numpy()
    
    grouped_class = grouped_class.reset_index()#we want the class_id columns as values, not as index
    for cid in particle_classes_id:
        df_cls = grouped_class[grouped_class["class_id"].astype(int) == int(cid)]#now class_id its a columns of the df and not an index or variable
        if df_cls.empty:#for the rest of the classes with N = 0
            continue
        # velocities cm segregated by classes
        df_cls["p_x"] = (df_cls["p_x"] / df_cls["mass"]).to_numpy()
        df_cls["p_y"] = (df_cls["p_y"] / df_cls["mass"]).to_numpy()
        df_cls["p_sp"]  = (df_cls["p_sp"] / df_cls["mass"]).to_numpy()
        df_cls["class_id"] = cid
        df_cls["p_x"] = apply_savgol_filter(df_cls["p_x"] , window_length, polyorder)
        df_cls["p_y"] = apply_savgol_filter(df_cls["p_y"], window_length, polyorder)
        df_cls["p_sp"] = apply_savgol_filter(df_cls["p_sp"], window_length, polyorder)
        dfs_block_segregated[cid].append(df_cls[df_cls['frame_id'] < max_frames])#save df
        
    #time vector
    time = np.linspace(0, max_time, len(grouped["frame_id"]))

    #smoothed variables with the SG filter
    grouped["p_x"] = apply_savgol_filter(grouped["p_x"],window_length,polyorder)
    grouped["p_y"] = apply_savgol_filter(grouped["p_y"],window_length,polyorder)
    grouped["p_sp"] = apply_savgol_filter(grouped["p_sp"],window_length,polyorder)
    grouped["speed_mean"] = apply_savgol_filter(grouped["speed_mean"],window_length,polyorder)
    grouped["vx_mean"] = apply_savgol_filter(grouped["vx_mean"],window_length,polyorder)
    grouped["vy_mean"] = apply_savgol_filter(grouped["vy_mean"],window_length,polyorder)
    dfs_block.append(grouped[grouped['frame_id'] < max_frames])#save df
     
    # creates a dataFrame with all the values
    for i in range(len(all_particles_speeds)):
        id_particle = i
        clean_speeds_1p = eliminate_vel(np.array(all_particles_speeds[i]),THRES)
        clean_vx_1p = eliminate_vel(np.array(all_particles_vx[i]),THRES)
        clean_vy_1p = eliminate_vel(np.array(all_particles_vy[i]),THRES)
        tem_aveg_speed = temporal_prom(clean_speeds_1p)
        tem_aveg_vx = temporal_prom(clean_vx_1p)
        tem_aveg_vy = temporal_prom(clean_vy_1p)
        p_class = particles_class[i]
        speed_particles.append(tem_aveg_speed)
        vx_particles.append(tem_aveg_vx)
        vy_particles.append(tem_aveg_vy)
        for h in range(1,len(particle_type_num)+1):#loop in particle class types
            if p_class == particle_classes_id[h-1]:
                speed_particles_segregated[h].append(tem_aveg_speed)
                vx_particles_segregated[h].append(tem_aveg_vx)
                vy_particles_segregated[h].append(tem_aveg_vy)
    #save data to do de repetitions mean
    for c_idx in range(1,len(particle_type_num)+1):
        sp_block_segregated[c_idx].append(speed_particles_segregated[c_idx])#every list value is for a video_id
        vx_block_segregated[c_idx].append(vx_particles_segregated[c_idx])
        vy_block_segregated[c_idx].append(vy_particles_segregated[c_idx])
                
    count_classes = np.array([
            particles_class.count(1),
            particles_class.count(2),
            particles_class.count(3),
            ])
    
    #save data to do de repetitions mean
    sp_block.append(speed_particles)#every list value is for a video_id
    vx_block.append(vx_particles)
    vy_block.append(vy_particles)
    
    eliminated = np.zeros(3)#WE ARE NOT FILTERING PARTICLES
    control_particles(count_classes,particle_type_num,eliminated) 
    vectx_speed, vecty_speed = get_histogram_data(speed_particles, num_bins=num_bins)
    vectx_vx, vecty_vx = get_histogram_data(vx_particles, num_bins=num_bins)
    vectx_vy, vecty_vy = get_histogram_data(vy_particles, num_bins=num_bins)

    #smooth continuos x values:
    if len(speed_particles) > 0:
        smooth_vectx_speed = np.linspace(min(speed_particles), max(speed_particles), 200)
        smooth_vectx_vx = np.linspace(min(vx_particles), max(vx_particles), 200)
        smooth_vectx_vy = np.linspace(min(vy_particles), max(vy_particles), 200)
    else:
        print("speed_particles lenght = 0")

dfs_total = pd.concat(dfs_block, ignore_index=True)#df with ALL the dfs
existing_videos = sorted(list(dfs_total["video_id"].unique()))#only video_ids column, list of indexs with not df format (we use list function)
total_videos = len(existing_videos)
print("Last video in dfs_total:", existing_videos[-1])
#print("Total de vídeos reales encontrados:", len(existing_videos))
for index in range(0, len(existing_videos), n_same_vid):# index values: 1, 5, 10, 15...
    end_num = min(index + n_same_vid - 1, N_videos)# Theoretical end_in
    expected_ids = [f"{i:04d}" for i in range(index, end_num + 1)]
    block_ids = existing_videos[index : index + n_same_vid]
    if cid >= len(dfs_block) or not block_ids:# from video 11 to 15, eliminated because there are only plastics
        continue
    df_block = dfs_total[dfs_total["video_id"].isin(block_ids)]
    df_mean_5v = df_block.groupby("frame_id")[["p_x", "p_y", "p_sp", "speed_mean",  "vx_mean", "vy_mean"]].mean().reset_index()#df index go to 0 to 4
    Nmet = df_block["Nmet"].iloc[0]
    #df_std_5v = df_block.groupby("frame_id")[["p_x", "p_y", "p_sp"]].std().reset_index()
    start_ind = block_ids[0]
    idx_start = index
    idx_end = index + len(block_ids) #asure idx_start and idx_end are not the same
    end_ind = block_ids[-1]
    df_config = df_config.reset_index(drop=True)
    NsP = int(df_config.loc[idx_start, "N_Small_Plastics"])
    NbP = int(df_config.loc[idx_start, "N_Big_Plastics"])
    pf_met = Nmet*get_class_area(1,class_radius)/Area_box
    if index == idx_start:
        met_pf.append(pf_met)
    pf_sP = NsP*get_class_area(2,class_radius)/Area_box
    pf_bP = NbP*get_class_area(3,class_radius)/Area_box
    times = df_mean_5v["frame_id"]/30 #30 FPS
    df_mean_5v["time"] = times
    #save file with values to plot with Gnuplot
    output_file = f"{output_dir_cm}/Mean_Std_CM_videos_{start_ind}_to_{end_ind}.txt"#has the mean and the cm velocities data
    df_mean_5v.to_csv(output_file, sep=" ", index=False, float_format="%.4f")
    print(f"Procesed block: videos from {start_ind} to {end_ind}")
    
    #do the compleated histograms with all the particles of the 5 repetitions
    speed_5v = np.concatenate(sp_block[idx_start : idx_end])#1D vector
    vx_5v = np.concatenate(vx_block[idx_start : idx_end])
    vy_5v = np.concatenate(vy_block[idx_start : idx_end])
    
    #mean values and desviations
    speed_mean_5v = np.mean(speed_5v)
    speed_std_5v = np.std(speed_5v)
    vx_mean_5v = np.mean(vx_5v)
    vx_std_5v = np.std(vx_5v)
    vy_mean_5v = np.mean(vy_5v)
    vy_std_5v = np.std(vy_5v)
    
    #save in a list to plot later in gnuplot
    mean_val.append({
        "Pfmet": pf_met,
        "speed_mean":speed_mean_5v,
        "speed_des":speed_std_5v,
        "vx_mean": vx_mean_5v,
        "vx_des": vx_std_5v,
        "vy_mean": vy_mean_5v,
        "vy_des": vy_std_5v,
        })
    
    #Do the histograms 
    #Vx histogram
    plt.figure()
    plt.title(r'$v_x$ distribution ($\phi_p=0.30, \phi_m=0.05$)', fontsize=12)
    plt.xlabel(r'$v_x$ [cm/s]')
    plt.ylabel(r'$v_x$ Counts')
    counts, bins, patches = plt.hist(vx_5v, bins = num_bins, edgecolor="black", color="green", label=rf'<$v_x$> = {vx_mean_5v:.3f}, $\sigma$ = {vx_std_5v:.3f}')
    indice_max = np.argmax(counts)
    vx_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
    plt.axvline(x=vx_freq, color="black", linestyle="--", label = rf"Peak frequency for $v_x$ = {vx_freq:.1f} [cm/s]")
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir_dist}/Vx_Distr_{pf_sP:.2f}_{pf_met:.2f}.png", dpi=300)
    plt.close()
    
    #Vy histogram
    vy_freq = np.argmax(vy_5v)
    plt.figure()
    plt.title(r'$v_y$ distribution ($\phi_p=0.30, \phi_m=0.05$)', fontsize=12)
    plt.xlabel(r'$v_y$ [cm/s]')
    plt.ylabel(r'$v_y$ Counts')
    counts, bins, patches = plt.hist(vy_5v, bins = num_bins, edgecolor="black", color="green", label=rf'<$v_y$> = {vy_mean_5v:.3f}, $\sigma$ = {vy_std_5v:.3f}')
    indice_max = np.argmax(counts)
    vy_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
    plt.axvline(x=vy_freq, color="black", linestyle="--", label = rf"Peak frequency for $v_y$ = {vy_freq:.1f} [cm/s]")
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir_dist}/Vy_Distr_{pf_sP:.2f}_{pf_met:.2f}.png", dpi=300)
    plt.close()
    
    #Speed histogram
    speed_freq = np.argmax(speed_5v)
    plt.figure()
    plt.title(r'Speed distribution ($\phi_p=0.30, \phi_m=0.05$)', fontsize=12)
    plt.xlabel("Speed [cm/s]")
    plt.ylabel("Speed Counts")
    counts, bins, patches = plt.hist(speed_5v, bins = num_bins, edgecolor="black", color="green", label=rf'<speed> = {speed_mean_5v:.3f}, $\sigma$ = {speed_std_5v:.3f}')
    indice_max = np.argmax(counts)
    speed_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
    plt.axvline(x=speed_freq, color="black", linestyle="--", label = f"Peak frequency for speed = {speed_freq:.1f} (cm/s)")
    plt.legend(loc="upper right")
    plt.savefig(f"{output_dir_dist}/Speed_Distr_{pf_sP:.2f}_{pf_met:.2f}.png", dpi=300)
    plt.close()

mean_val = pd.DataFrame(mean_val)
mean_val["day"] = day
cols_export = ["day", "Pfmet", "speed_mean", "speed_des", "vx_mean", "vx_des", "vy_mean", "vy_des"]
df_export = mean_val[cols_export]
df_export = df_export.sort_values(by=["Pfmet"])#increasing order in Nmet variable
df_export = df_export.sort_values(by=["day"])
#calculates +- of the mean values 
df_export["speed_des"] = df_export["speed_des"]/np.sqrt(n_same_vid)
df_export["vx_des"] = df_export["vx_des"]/np.sqrt(n_same_vid)
df_export["vy_des"] = df_export["vy_des"]/np.sqrt(n_same_vid)
#Save mean velocities and speed values (for all the experiments) in a text file to plot in Gnuplot
c3 = f"C:/Users/Neyda/Documents/Materia_Activa_UNAV/Outputs/d_{distance}/Mean_vel_all_experiments.txt"
file_exists = os.path.exists(c3)
with open(c3, "a") as f:
    if not file_exists:
        f.write("day Pfmet mean_sp(cm/s) std_sp(cm/s)  mean_vx(cm/s) std_vx(cm/s) mean_vy(cm/s) std_vy(cm/s)\n")
    df_export.to_csv(f, sep=" ", header=False, index=False, float_format="%.4f")
    f.write("\n")
    f.write("\n")
    
df_config = df_config.reset_index()

#Segregated by classes 
for cid in particle_classes_id:
    if cid >= len(dfs_block_segregated) or not dfs_block_segregated[cid]:#check list lenth
        continue
    dfs_total_s = pd.concat(dfs_block_segregated[cid], ignore_index=True)#df with ALL the dfs
    total_videos_s = len(existing_videos)
    class_ident = get_class_name(cid)
    for index in range(0, len(existing_videos), n_same_vid):# index values: 1, 5, 10, 15...
        end_num = min(index + n_same_vid - 1, N_videos)# Theoretical end_in
        expected_ids = [f"{i:04d}" for i in range(index, end_num + 1)]
        block_ids_s = existing_videos[index : index + n_same_vid]
        if not block_ids_s:# from video 11 to 15, eliminated because there are only plastics
            continue
        df_block_s = dfs_total_s[dfs_total_s["video_id"].isin(block_ids_s)]
        df_mean_5v_s = df_block_s.groupby("frame_id")[["p_x", "p_y", "p_sp"]].mean().dropna().reset_index()#df index go to 0 to 4
        first_video = block_ids_s[0]#first video of the block
        start_ind = block_ids_s[0]
        idx_start = existing_videos.index(start_ind)
        idx_end = idx_start + len(block_ids_s)
        end_ind = block_ids_s[-1]
        #HERE IT WOULD BE NECCESARY AND AUTOMAT. IN THE FUTURE
        Nmet = int(df_config.loc[idx_start, "N_Metallics"])
        NsP = int(df_config.loc[idx_start, "N_Small_Plastics"])
        NbP = int(df_config.loc[idx_start, "N_Big_Plastics"])
        pf_met = Nmet*get_class_area(1,class_radius)/Area_box
        pf_sP = NsP*get_class_area(2,class_radius)/Area_box
        pf_bP = NbP*get_class_area(3,class_radius)/Area_box
        Ntotal = Nmet + NsP + NbP
        times = df_mean_5v_s["frame_id"]/30 #30 FPS
        df_mean_5v_s["time"] = times
        df_mean_5v_s["abs(p_y)"] = abs(df_mean_5v_s["p_y"])
        #Save mean velocities and speed values (for all the experiments) in a text file to plot in Gnuplot
        c4 = f"{output_dir_cm}/Mean_Std_CM_videos_{start_ind}_to_{end_ind}_class={class_ident}.txt"
        with open(c4, "w") as f:
            f.write("frame_id p_vx p_vy p_sp time abs(p_y)\n")
            df_mean_5v_s.to_csv(f, sep=" ", header=False, index=False, float_format="%.4f")
            print(f"Procesed block: videos from {start_ind} to {end_ind} for class = {class_ident}")
        
        #do the compleated histograms with all the particles of the 5 repetitions
        ind_bloc = [existing_videos.index(v_id) for v_id in block_ids_s if v_id in existing_videos]#if there is a video with only plastics, segregated list for mateallics indexs dont match with the plastics ones
        vx_list_block = [vx_block_segregated[cid][i] for i in ind_bloc if i < len(vx_block_segregated[cid])]#v_id must be in the in the block of five we are analysing and in the total existing videos for the particle type class
        vy_list_block = [vy_block_segregated[cid][i] for i in ind_bloc if i < len(vy_block_segregated[cid])]#we get the real position of the video_id we want in the existing_video list for that particle type class
        sp_list_block = [sp_block_segregated[cid][i] for i in ind_bloc if i < len(sp_block_segregated[cid])]
        
        # concatenate list if they are not empty for the five repetitions
        vx_5v_s = np.concatenate([np.array(v) for v in vx_list_block if len(v) > 0]) if any(len(v) > 0 for v in vx_list_block) else np.array([])
        vy_5v_s = np.concatenate([np.array(v) for v in vy_list_block if len(v) > 0]) if any(len(v) > 0 for v in vy_list_block) else np.array([])
        speed_5v_s = np.concatenate([np.array(v) for v in sp_list_block if len(v) > 0]) if any(len(v) > 0 for v in sp_list_block) else np.array([])
            
        plot_color = class_colors.get(cid, "gray")
            #Do the histograms 
            
            #Vx histogram
        if len(vx_5v_s) > 0:
            vx_mean_5v = np.mean(vx_5v_s)
            vx_mean_den[cid].append(vx_mean_5v)
            vx_std_5v = np.std(vx_5v_s)
            vx_freq = np.argmax(vx_5v_s) 
            plt.figure()
            plt.title(rf"$v_x$ {class_ident} particle's distribution ($\phi_p=0.30, \phi_m=0.05$)", fontsize=12)
            plt.xlabel(r'$v_x$ [cm/s]')
            plt.ylabel(r'$v_x$ Counts')
            counts, bins, patches = plt.hist(vx_5v_s, bins = num_bins, edgecolor="black", color=plot_color, label=rf' <$v_x$>= {vx_mean_5v:.3f}, $\sigma$ = {vx_std_5v:.3f}')
            indice_max = np.argmax(counts)
            vx_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
            plt.axvline(x=vx_freq, color="black", linestyle="--", label = rf"Peak frequency for $v_x$ = {vx_freq:.1f} [cm/s]")
            plt.legend(loc="upper right")
            plt.savefig(f"{output_dir_dist}/Vx_Distr_{pf_sP:.2f}_{pf_met:.2f}_class={class_ident}.png", dpi=300)
            plt.close()
            
        #Vy histogram
        if len(vy_5v_s) > 0:
            vy_mean_5v = np.mean(vy_5v_s) 
            vy_std_5v = np.std(vy_5v_s)
            vy_freq = np.argmax(vy_5v_s) 
            plt.figure()
            plt.title(rf"$v_y$ {class_ident} particle's distribution ($\phi_p=0.30, \phi_m=0.05$)", fontsize=12)
            plt.xlabel(r'$v_y$ [cm/s]')
            plt.ylabel(r'$v_x$ Counts')
            counts, bins, patches = plt.hist(vy_5v_s, bins = num_bins, edgecolor="black", color=plot_color, label=rf'<$v_y$> = {vy_mean_5v:.3f}, $\sigma$ = {vy_std_5v:.3f}')
            indice_max = np.argmax(counts)
            vy_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
            plt.axvline(x=vy_freq, color="black", linestyle="--", label = rf"Peak frequency for $v_y$ = {vy_freq:.1f} [cm/s]")
            plt.legend(loc="upper right")
            plt.savefig(f"{output_dir_dist}/Vy_Distr_{pf_sP:.2f}__{pf_met:.2f}class={class_ident}.png", dpi=300)
            plt.close()
        
        #Speed histogram
        if len(speed_5v_s) > 0:
            speed_mean_5v = np.mean(speed_5v_s) 
            speed_std_5v = np.std(speed_5v_s)
            speed_freq = np.argmax(speed_5v_s) 
            plt.figure()
            plt.title(rf"Speed {class_ident} particle's distribution ($\phi_p=0.30, \phi_m=0.05$)", fontsize=12)
            plt.xlabel("Speed [cm/s]")
            plt.ylabel("Speed Counts")
            counts, bins, patches = plt.hist(speed_5v_s, bins = num_bins, edgecolor="black", color=plot_color, label=rf'<speed> = {speed_mean_5v:.3f}, $\sigma$ = {speed_std_5v:.3f}')
            indice_max = np.argmax(counts)
            speed_freq = (bins[indice_max] + bins[indice_max + 1]) / 2.0
            plt.axvline(x=speed_freq, color="black", linestyle="--", label = f"Peak frequency for speed = {speed_freq:.1f} [cm/s]")
            plt.legend(loc="upper right")
            plt.savefig(f"{output_dir_dist}/Speed_Distr_{pf_sP:.2f}__{pf_met:.2f}class={class_ident}.png", dpi=300)
            plt.close()
            
            
#Nmet/Ntotal vs mean velocities graphics
print("Metallic packing fractions:")
data_den = {"met pf": pd.Series(met_pf)}#dictionary so we can change columns number regarding the part. class types existing
for i, name in enumerate(particle_type_class):
    data_den[f"vx_mean_{name}"] = pd.Series(vx_mean_den.get(i+1, []))
df_den = pd.DataFrame(data_den)
#df_den["Nmet"] = N_met_list
df_den = df_den.sort_values(by=["met pf"]).reset_index(drop=True)#order data regarding Nmet/Ntotal
c5 = f"{output_dir_dist}/Vx_Met_densities_d_{distance}_day_{day}.txt"
with open(c5, "w") as f:
    f.write(" ".join(df_den.columns) + "\n")  #headline with the name of the diff class types
    df_den.to_csv(f, sep=" ", header=False, index=False, float_format="%.4f",na_rep="?")#change nan to ? so gnuplot can read it
            
                
            