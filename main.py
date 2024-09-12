import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd 
from dbfread import DBF
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
#TODO append you system path to start the programm
sys.path.append(r'C:\Users\User\Desktop\bsc-Coding')
from network import Network
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class main:

    #np.random.seed(42)

    #hyperparameters
    input_len=5
    layer_len=[5,64,32,1]
    epochs=200
    learningrate=0.005
    batch_size=256
    k = 5



    
    # Specify the path to your DBF files
    dbf_paths = [
        r'C:\Users\User\Desktop\bsc-Coding\Training_August312016_ExportTable_ExportTable.dbf',
        r'C:\Users\User\Desktop\bsc-Coding\Test_August312016_ExportTable_ExportTable.dbf',
    ]

# Load each DBF file into a DataFrame
    dataframes = []
    for dbf_path in dbf_paths:
        table = DBF(dbf_path)
        
        df = pd.DataFrame(iter(table))
        print(dbf_path)
        print(df)
        dataframes.append(df)

# Combine all DataFrames into one
    data = pd.concat(dataframes, ignore_index=True)


# Print the combined DataFrame
    print(data.head())
    print(data)
    
    # Print the records (rows) of the DBF file
    print("\nRecords:")
    

    #explanatory Variables used for FCDNN 
    predictVars = ['BAND_1', 'BAND_2', 'BAND_3', 'BAND_4'] 

    #response Variable (1= has SDS, 0= does not have SDS)
    responseVar=['GRNDTRUTH']


    columnNames=predictVars+responseVar
    
    #rename columns for readablility
    new_names = {'BAND_1': 'Blue', 'BAND_2': 'Green', 'BAND_3': 'Red', 'BAND_4': 'NIR'}
    data.rename(columns=new_names, inplace=True)
    print(data.head())


    # Calculate NDVI and put it in a new column
    ndvi = (data["NIR"] - data["Red"]) / (data["NIR"] + data["Red"])
    data.insert(5, 'NDVI', ndvi)
    print(data)


    expl_vars = ['Blue', 'Green', 'Red', 'NIR', 'NDVI' ]
    resp_var = "GRNDTRUTH"

    X = data[expl_vars] 
    Y = data[resp_var]
    
   

    print(X)

    #print("this is before split: ")
    #print(X)
    X=np.asarray(X,dtype=float)
    Y=np.asarray(Y,dtype=float)

    Y= Y.reshape(-1,1)
    print("shape of y:",Y.shape)
   
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
   
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

# Apply normalization to training data
    X_train_normalized = (X_train - X_train_mean) / X_train_std
    print(X_train_normalized)

# Apply the same normalization parameters to test data
    X_test_normalized = (X_test - X_train_mean) / X_train_std

    '''
# Checking the statistics of the normalized data
    print("Statistics of normalized training data:")
    print("Mean:", np.mean(X_train_normalized, axis=0))
    print("Std Dev:", np.std(X_train_normalized, axis=0))

# Ensuring no extreme values
    print("Min values:", np.min(X_train_normalized, axis=0))
    print("Max values:", np.max(X_train_normalized, axis=0))

# Inspecting the shape and content of the data
    print("Shape of X_train_normalized:", X_train_normalized.shape)
    print("Sample of normalized data:", X_train_normalized[:5])
    '''   


    #start and initialize algorithm 
    dropout_rate=[0.1,0.1,0.2,0.0]
    network= Network(input_len=input_len, output_len=layer_len,dropout_rates=dropout_rate)
    losses=network.train_network(learning_rate=learningrate,X_train=X_train_normalized,Y_train=y_train,epoch_n=epochs,batch_size=batch_size)

    '''  
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()
    print("after training:")
    def prediction(network,row):
       return network.forwardPropagation(row)
    
    
    for i in range(len(X_test_normalized)):
        row = X_test_normalized[i]
        row=row.reshape(1,-1)
        #print("row shape:",row.shape)
        #print("row values:",row)
        pred = prediction(network=network, row=row)

        print(f"Expected={y_test[i]}, Actual={pred}")


    test_predictions = network.forwardPropagation(X_test_normalized)
    test_predictions_binary = (test_predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, test_predictions_binary)
    precision = precision_score(y_test, test_predictions_binary)
    recall = recall_score(y_test, test_predictions_binary)
    f1 = f1_score(y_test, test_predictions_binary)


    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    
    cm = confusion_matrix(y_true=y_test, y_pred=test_predictions_binary)

    # Plotting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


    fpr, tpr, _ = roc_curve(y_true=y_test, y_score=test_predictions)  # false positive rate, true positive rate
    roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


'''

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Store the metrics across all folds
    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

# Start k-fold cross-validation
    for train_index, val_index in kf.split(X_train_normalized):
        X_train_fold, X_val_fold = X_train_normalized[train_index], X_train_normalized[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Initialize the network
        dropout_rate = [0.1, 0.1, 0.2, 0.0]
        network = Network(input_len=input_len, output_len=layer_len, dropout_rates=dropout_rate)

    # Train the network
        losses = network.train_network(learning_rate=learningrate, X_train=X_train_fold, Y_train=y_train_fold, epoch_n=epochs, batch_size=batch_size)
        '''
    # Plot the loss curve for this fold
        plt.plot(range(epochs), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss over Epochs (Fold)')
        plt.show()
        '''
    # Evaluate the network on the validation set
        val_predictions = network.forwardPropagation(X_val_fold)
        val_predictions_binary = (val_predictions > 0.5).astype(int)

        accuracy = accuracy_score(y_val_fold, val_predictions_binary)
        precision = precision_score(y_val_fold, val_predictions_binary)
        recall = recall_score(y_val_fold, val_predictions_binary)
        f1 = f1_score(y_val_fold, val_predictions_binary)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # Compute ROC AUC
    fpr, tpr, _ = roc_curve(y_true=y_val_fold, y_score=val_predictions)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # Optionally, plot the confusion matrix and ROC curve for this fold
    cm = confusion_matrix(y_true=y_val_fold, y_pred=val_predictions_binary)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    print(f'Length of accuracies: {len(accuracies)}')
    print(f'Length of precisions: {len(precisions)}')
    print(f'Length of recalls: {len(recalls)}')
    print(f'Length of f1s: {len(f1s)}')
    print(f'Length of aucs: {len(aucs)}')
    # Create a DataFrame to summarize the results
    results_df = pd.DataFrame({
        'Fold': range(1, k + 1),
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1s,
    })

# Compute the average metrics across all folds
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_auc = np.mean(aucs)
    print(f'Length of accuracies: {len(accuracies)}')
    print(f'Length of precisions: {len(precisions)}')
    print(f'Length of recalls: {len(recalls)}')
    print(f'Length of f1s: {len(f1s)}')
    print(f'Length of aucs: {len(aucs)}')

    print(f'Average Accuracy: {avg_accuracy}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F1 Score: {avg_f1}')
    print(f'Average ROC AUC: {avg_auc}')
    # Append average metrics to the DataFrame
    averages =pd.Series({
        'Fold': 'Average',
        'Accuracy': avg_accuracy,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1 Score': avg_f1,
    })
    results_df = pd.concat([results_df, averages.to_frame().T], ignore_index=True)
    print(results_df)
