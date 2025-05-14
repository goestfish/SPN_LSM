from huggingface_hub import login
from huggingface_hub import whoami
from datasets import load_dataset
from datasets import load_from_disk
from datasets import  concatenate_datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import cv2  # Only used for CLAHE
import random
import pandas as pd
from collections import defaultdict
from datasets import Dataset, DatasetDict

# Step 1: Extract patient IDs from the 'Path' column
def extract_patient_id(path):

    return path.split('/')[2]

def assign_split(example,train_patients,val_patients,test_patients):
    patient_id = example['patient_id']
    if patient_id in train_patients:
        return 'train'
    elif patient_id in val_patients:
        return 'validation'
    elif patient_id in test_patients:
        return 'test'

def load_filter_data_once():
    datasetDict = load_from_disk(chexpert_data_path)
    print(datasetDict.column_names['train'])
    print('wuhu')

    # Assuming datasetDict is your original dataset
    # Filter the 'train' split
    print(datasetDict['train'][:20])
    train_filtered_0 = datasetDict['train'].filter(
        lambda x: x['Frontal/Lateral'] == 0 and  x['No Finding'] == 3)
    train_filtered = datasetDict['train'].filter(
        lambda x: x['Frontal/Lateral'] == 0 and  x['Cardiomegaly'] == 3)

    validation_filtered_0 = datasetDict['validation'].filter(lambda x: x['Frontal/Lateral'] == 0 and  x['No Finding'] == 3)
    # Filter the 'validation' split
    validation_filtered = datasetDict['validation'].filter( lambda x: x['Frontal/Lateral'] == 0 and x['Cardiomegaly'] == 3)
    print(len(train_filtered_0),len(train_filtered),len(validation_filtered_0),len(validation_filtered))

    # Merge 'train' and 'validation' splits
    merged_dataset = concatenate_datasets([train_filtered, validation_filtered,train_filtered_0,validation_filtered_0])

    merged_dataset.save_to_disk(chexpert_data_path_filter)

# Function to load an image as a NumPy array
def load_image_as_array(example):
    image = Image.open(example['Path']).convert("RGB")  # Ensure it's RGB
    example['image_array'] = np.array(image)
    return example


def make_train_test_val(ds,seed=42):
    # Add a new column to store patient IDs
    ds = ds.map(lambda example: {'patient_id': extract_patient_id(example['Path'])})

    # Step 2: Group data by patient IDs
    patient_to_indices = defaultdict(list)
    for i, example in enumerate(ds):
        patient_to_indices[example['patient_id']].append(i)

    # Step 3: Split patient IDs into train, validation, and test
    patient_ids = list(patient_to_indices.keys())
    random.seed(seed)  # For reproducibility
    random.shuffle(patient_ids)

    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate split indices
    num_patients = len(patient_ids)
    train_end = int(train_ratio * num_patients)

    # Split the patient IDs
    train_patients_tmp = patient_ids[:train_end]
    test_patients = set(patient_ids[train_end:])

    # Further split train into train and validation
    random.seed(0)
    random.shuffle(train_patients_tmp)

    val_end = int(len(train_patients_tmp) * val_ratio)
    val_patients = set(train_patients_tmp[:val_end])
    train_patients = set(train_patients_tmp[val_end:])

    # Add a split column to the dataset
    ds = ds.map(lambda example: {'split': assign_split(example, train_patients, val_patients, test_patients)})

    # Step 5: Split the dataset into train/validation/test
    split_datasets = DatasetDict({
        split: ds.filter(lambda example: example['split'] == split)
        for split in ['train', 'validation', 'test']
    })

    # Step 6: Ensure 50-50 balance of 'No Finding' == 3 in each split
    new_dict=dict()
    for split in ['train', 'validation', 'test']:


        no_finding_3 = split_datasets[split].filter(lambda x: x['No Finding'] == 3)
        other_findings = split_datasets[split].filter(lambda x: x['No Finding'] != 3)

        # Determine the target size (50% of the smaller class)
        target_size = min(len(no_finding_3), len(other_findings))
        print(len(no_finding_3), len(other_findings))

        # Randomly sample to achieve balance
        no_finding_3_balanced = no_finding_3.shuffle(seed=42).select(range(target_size))
        other_findings_balanced = other_findings.shuffle(seed=42).select(range(target_size))
        print(len(no_finding_3_balanced), len(other_findings_balanced))
        # Combine balanced datasets
        balanced_df =concatenate_datasets([no_finding_3_balanced,other_findings_balanced])
        new_dict[split]=balanced_df


    return new_dict



def process_image(pil_image, size=(128, 128)):


    # Step 1: Convert PIL image to NumPy array (OpenCV format)
    pil_img = pil_image.convert("RGB")  # Ensure it is in RGB format
    img = np.array(pil_img)

    # Step 2: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Note: RGB instead of BGR

    # Step 3: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # Step 4: Resize the image to 256x256
    resized_img = cv2.resize(clahe_img, size, interpolation=cv2.INTER_AREA)
    return resized_img
def convert_to_numpy_data_and_save(ds, save_path):

    save_data={}

    for data_type,dataset_row in ds.items():
        #print(data_type,dataset)

        # Initialize a batch container
        X = []
        int_data=[]
        float_data=[]

        for record in dataset_row:
            # Convert the 'image' column to a NumPy array
            image = record['image']
            #print(image)
            image=process_image(image)
            #print(image.shape)

            # Append the image to the batch
            X.append(image)
            if record['No Finding']==3:
                y=0
            else:
                y=1
            int_data.append([y,record['Sex'],record['AP/PA']])
            float_data.append([record['Age']])

        X= np.stack(X)

        ###
        #preprocessing goes here:


        ####
        int_data = np.stack(int_data)
        float_data = np.stack(float_data)
        save_data[data_type]=[X,int_data,float_data]
        print(np.mean(int_data,axis=0))
    pickle.dump(save_data,open(save_path,'wb'))


def short_analysis(save_path):
    save_data=pickle.load(open(save_path,'rb'))
    for key, val in save_data.items():
        [X, int_data, float_data]=val
        print(int_data.shape)
        print('percentage cardio',np.mean(int_data[:,0]))



if __name__ == '__main__':
    # Prompt user for Hugging Face token
    hf_token = input("Please enter your Hugging Face access token: ")
    login(token=hf_token)
    user_info = whoami()
    ds = load_dataset("danjacobellis/chexpert")
    
    # Save the dataset to disk
    chexpert_data_path="./data_chexpert/"
    chexpert_data_path_filter="./data_chexpert_filter/"
    ds.save_to_disk(chexpert_data_path)
    
    load_filter_data_once()

    ds = load_from_disk(chexpert_data_path_filter)
    print('split')
    filter_datasets=make_train_test_val(ds)
    save_path='chexpert.pkl'
    print('save')
    convert_to_numpy_data_and_save(filter_datasets,save_path)

    #save_path= '../chexpert.pkl'
    short_analysis(save_path)

    '''
    merged_dataset = load_from_disk(chexpert_data_path_filter)


    # Split the merged dataset into train, validation, and test
    train_val, test = train_test_split(merged_dataset, test_size=0.2, random_state=42)
    train, validation = train_test_split(train_val, test_size=0.25, random_state=42)  # 20% of original = 80% * 0.25
    
    # Display the sizes of the splits
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(validation)}")
    print(f"Test size: {len(test)}")
    
    with_array=merged_dataset.map(load_image_as_array)
    image_arrays = np.stack(with_array['image_array'])
    '''
