import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
def load_data(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_data(point_clouds, mmwave_data, keypoints, output_file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point clouds (in blue)
    ax.scatter(point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2], c='blue', label='Point Clouds', s=1)

    # Plot mmwave data (in orange) if available
    if mmwave_data is not None and len(mmwave_data) > 0:
        ax.scatter(mmwave_data[:, 0], mmwave_data[:, 1], mmwave_data[:, 2], c='orange', label='MMWave Data', s=1)

    # Plot keypoints (in red) if available
    if keypoints is not None and len(keypoints) > 0:
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='red', label='Keypoints', s=5)

    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1)
    ax.set_zlim(4, 2)

    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-4, -2)
    # ax.set_zlim(-1, 1)

    # ax.set_xlim(2,4)
    # ax.set_ylim(-2,2)
    # ax.set_zlim(-1,1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Point Cloud and MMWave Data Visualization')
    ax.legend()

    # Save the plot as an image file
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory

def main():
    pkl_folder = '/home/ryan/MM-Fi/LEMT/data_dual/mmfi_dual.pkl'  # Change this to your folder path
    case_name = 'case_1'  # Specify the case you want to check
    output_folder = '/home/ryan/MM-Fi/PcDisplay'  # Change this to your desired output folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the .pkl file
    pkl_file = pkl_folder
    data = load_data(pkl_file)

    # Assuming data is structured as described
    sequences = data['sequences']

    # Select the first sequence for visualization
    if sequences:
        for sequence in sequences:
            if sequence.get('case_name') == case_name:
                break       
            point_clouds_list = sequence['point_clouds']
            mmwave_data_list = sequence['mmwave_data']
            keypoints_list = sequence.get('keypoints', [])
            # print("Number of frames in the sequence:", len(point_clouds_list))
            # Iterate through frames
            for frame_idx in range(len(point_clouds_list)):
                # print("Length of point clouds list:", len(point_clouds_list))
                # print("Index of current frame:", frame_idx)
                point_clouds = point_clouds_list[frame_idx]
                mmwave_data = mmwave_data_list[frame_idx] if frame_idx < len(mmwave_data_list) else None
                keypoints = keypoints_list[frame_idx] if frame_idx < len(keypoints_list) else None
                #filter out point clouds with y < -0.75
                point_clouds = point_clouds[point_clouds[:,1]>-0.75]
                # Convert to numpy arrays if needed
                point_clouds = np.array(point_clouds)
                if mmwave_data is not None:
                    mmwave_data = np.array(mmwave_data)
                if keypoints is not None:
                    keypoints = np.array(keypoints)
                #mmwave data x - 0.5, z - 0.25, y become negative
                # if mmwave_data is not None:
                #     mmwave_data[:, 0] = mmwave_data[:, 0] 
                #     mmwave_data[:, 2] = mmwave_data[:, 2] - 0.25
                #     mmwave_data[:, 1] = -mmwave_data[:, 1]

                #rotate point clouds and mmwave data(in format[x,y,z,doppler speed, intensity] along x axis by 90 degrees, then along y axis by -90 degrees
                # rotation_matrix = np.array([[1, 0, 0],
                #                             [0, 0, -1],
                #                             [0, 1, 0]])

                # rotation_matrix = np.array([[0, 0, 1],
                #                             [1, 0, 0],
                #                             [0, 1, 0]])    
                # rotation_matrix = np.array([[1, 0, 0],
                #                             [0, 1, 0],
                #                             [0, 0, 1]])
                # point_clouds = point_clouds.dot(rotation_matrix.T)
                # if mmwave_data is not None:
                #     mmwave_data[:, :3] = mmwave_data[:, :3].dot(rotation_matrix.T)
                # if keypoints is not None:
                #     keypoints = keypoints.dot(rotation_matrix.T)
                # Define output file name
                output_file = os.path.join(output_folder, f"{case_name}_frame_0.png")
                # print("Output file path:", output_file)
                # Visualize and save the frame's data
                # print("point_clouds shape:", point_clouds.shape)
                # if mmwave_data is not None:
                    # print("mmwave_data shape:", mmwave_data.shape)
                plot_data(point_clouds, mmwave_data, keypoints, output_file)
                # print(f"Saved plot for frame {frame_idx} as {output_file}")
                time.sleep(0.1)
                # Remove this break if you want to visualize all frames
                  

if __name__ == "__main__":
    main()