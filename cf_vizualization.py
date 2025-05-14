import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


def plot_imgs(dataset_name,save_folder,fold_idx,model_title,gamma_beta,img_name,img_idx):
    gamma,beta=gamma_beta
    fig_size=2
    fig, axis = plt.subplots(2, len(model_title), figsize=(fig_size * len(model_title), fig_size * 2))
    row_titles = ['SPN', 'MLP']  # Labels for the rows
    for j,(vae_model_name,title) in enumerate(model_title):
        for i ,model_name in enumerate(row_titles):
            data_path_grid = save_folder +vae_model_name + '/grid' + str(0) + '/'+ 'fold_' + str(fold_idx) + '/'+ 'counterfactual_imgs/'
            save_file_path = data_path_grid+ 'datab' + str(beta).replace('.', '-') + 'g' + str(gamma).replace('.', '-') + model_name + '_1000.pkl'
            all_data_list = pkl.load(open(save_file_path, 'rb'))
            all_data = []
            for entry in all_data_list:
                all_data.extend(entry)

            [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss, log_pred, p_z, label_switch_step, y_cf_goal,
             coordinates, x_org, mean_cls, y_org] = zip(*all_data)
            x_cf = np.asarray(x_cf)
            x_rec = np.asarray(x_rec)
            x_org = np.asarray(x_org)

            differences_mean = np.mean(x_cf[img_idx]- x_rec[img_idx], axis=0)
            #axis[i,j].imshow(x_cf[img_idx][0, :, :, 0], cmap='gray')
            axis[i,j].imshow(differences_mean, cmap='jet', alpha=0.5)
            if not i:
                #mini_title=title+'\n'+model_name
                axis[i, j].set_title(title)


            axis[i,j].axis('off')
    # Add row titles on the left with vertical orientation
    for i, row_title in enumerate(row_titles):
        fig.text(0.015, 0.75 - (i * 0.5), row_title, va='center', ha='center',
                 rotation=90, fontsize=12, fontweight='bold')

    plt.tight_layout()
    print('save', img_name)
    plt.savefig(img_name)








if __name__ == '__main__':
    work_from = 0
    dataset_name=['CXR_process_org_distr','chexpert'][1]
    fold_idx = 0
    save_folder =['../model_results/',
                  '/lustre/miifs01/project/m2_datamining/sync_data/cnn_spn_models/' ][work_from]
    model_title=[('Chex_KLD1_le4_g2_64',r'$\beta_{1}=0.1000$'),
                 ('Chex_KLD2_le4_g2_64', r'$\beta_{1}=0.0100$'),
                 ('Chex_KLD3_le4_g2_64', r'$\beta_{1}$=0.0010'),
                 ('Chex_KLD4_le4_g2_64', r'$\beta_{1}$=0.0001'),
                 ]

    gamma_beta=(0,0)
    for img_idx in range(5):
        img_name = 'cf_viz_'+str(img_idx)+'_org_cf.png'
        plot_imgs(dataset_name,save_folder,fold_idx,model_title,gamma_beta,img_name,img_idx)