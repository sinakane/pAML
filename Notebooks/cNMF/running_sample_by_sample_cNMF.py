import os
from pathlib import Path
import muon as mu
import scanpy as sc
import pandas as pd
from cnmf import cNMF
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard

def do_NMF(adata, K, numiter, numhvgenes, wdir, sample_name):
    """
    Perform NMF analysis on the provided anndata object

    Args:
        adata: AnnData object
        K: Number of components
        numiter: Number of iterations
        numhvgenes: Number of highly variable genes
        wdir: Working directory path
        sample_name: Name of the sample

    Returns:
        tuple: (topgenes, usage_norm)
    """
    # Set raw counts
    adata.X = adata.layers["raw_counts"]

    # Remove any existing Usage columns
    usage_cols = [col for col in adata.obs.columns if col.startswith('Usage_')]
    if usage_cols:
        print(f"Removing {len(usage_cols)} existing Usage columns")
        adata.obs = adata.obs.drop(columns=usage_cols)

    # Setup paths
    input_path = os.path.join(wdir, sample_name, "NMF_K10")
    output_directory = os.path.join(wdir, sample_name, "NMF_K10", "out")
    results_directory = os.path.join(wdir, sample_name, "NMF_K10", "results")

    # Create directories if they don't exist
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    # Create count data filename
    count_adat_fn = os.path.join(input_path, f"{sample_name}.h5ad")
    sc.write(count_adat_fn, adata)

    # Initialize and run cNMF
    cnmf_obj = cNMF(output_dir=output_directory, name="NMF")
    cnmf_obj.prepare(counts_fn=count_adat_fn,
                     components=K,
                     n_iter=numiter,
                     seed=14,
                     num_highvar_genes=numhvgenes)

    cnmf_obj.factorize(worker_i=0, total_workers=1)
    cnmf_obj.combine()
    cnmf_obj.consensus(k=K[0], density_threshold=0.1,
                      show_clustering=True,
                      close_clustergram_fig=False)

    # Load results
    usage_norm, gep_scores, gep_tpm, topgenes = cnmf_obj.load_results(K=K[0], density_threshold=0.1)

    # Rename usage columns to match topgenes numbering (1 to K instead of 0 to K-1)
    usage_norm.columns = [f'Usage_{i+1}' for i in range(usage_norm.shape[1])]

    # Rename topgenes columns to match usage columns
    if isinstance(topgenes, pd.DataFrame):
        # Get numeric columns
        numeric_cols = [col for col in topgenes.columns if str(col).isdigit()]
        # Create rename dictionary
        rename_dict = {col: f'Usage_{col}' for col in numeric_cols}
        # Rename columns
        topgenes = topgenes.rename(columns=rename_dict)

    # Add usage to adata object
    adata.obs = pd.merge(left=adata.obs, right=usage_norm,
                        how='left', left_index=True, right_index=True)

    # Calculate and plot similarities
    similarities = calculate_program_similarities(
        topgenes,
        usage_norm,
        sample_name,
        results_directory
    )

    # Save results
    topgenes_file = os.path.join(results_directory, f"{sample_name}_topgenes.csv")
    usage_file = os.path.join(results_directory, f"{sample_name}_usage.csv")
    adata_file = os.path.join(results_directory, f"{sample_name}_with_usage.h5ad")

    # Save files
    topgenes.to_csv(topgenes_file)
    usage_norm.to_csv(usage_file)
    adata.write(adata_file)

    # Create program scores plot (only once)
    plot_program_scores(adata, sample_name, results_directory)

    return topgenes, usage_norm


def plot_gene_correlation(adata, gene_list, layer="lognorm_counts", method='spearman', output_file=None):
    """
    Plots a correlation heatmap of selected genes from an AnnData object.
    """
    # Print initial gene list info
    print(f"\nInitial gene list length: {len(gene_list)}")
    print(f"First few genes: {list(gene_list)[:5]}")

    # Ensure genes exist in the dataset
    missing_genes = [gene for gene in gene_list if gene not in adata.var_names]
    if missing_genes:
        print(f"Warning: The following genes were not found in adata: {missing_genes}")
        gene_list = [gene for gene in gene_list if gene in adata.var_names]
        print(f"Number of genes after filtering: {len(gene_list)}")
        print(f"Remaining genes (first 5): {gene_list[:5]}")

    if not gene_list:
        raise ValueError("No valid genes found in adata.var_names.")

    # Print layer info
    print(f"\nUsing layer: {layer}")
    print(f"Available layers: {list(adata.layers.keys())}")

    # Extract expression data
    if layer:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in AnnData object.")
        data = adata[:, gene_list].layers[layer]
    else:
        data = adata[:, gene_list].X

    # Print data shape
    print(f"Data shape: {data.shape}")
    # Convert to DataFrame
    df = pd.DataFrame(data.toarray() if hasattr(data, "toarray") else data,
                     columns=gene_list, index=adata.obs_names)

    # Print correlation info
    corr_matrix = df.corr(method=method)
    print(f"Correlation matrix shape: {corr_matrix.shape}")

    # Plot heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f"Gene Correlation Heatmap ({method.capitalize()} Correlation)")

    if output_file:
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()

def correlation_heatmap(adata, obs_columns, method='spearman', output_file=None):
    """
    Generates a correlation heatmap for selected columns in adata.obs.

    Parameters:
        adata (AnnData): The input AnnData object.
        obs_columns (list): List of column names from adata.obs to compute correlations.
        method (str): Correlation method ('spearman', 'pearson', or 'kendall'). Default is 'spearman'.
        output_file (str, optional): If specified, saves the plot to this file path.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """
    if not set(obs_columns).issubset(adata.obs.columns):
        missing_cols = set(obs_columns) - set(adata.obs.columns)
        raise ValueError(f"Columns {missing_cols} not found in adata.obs")

    # Extract relevant data
    df = adata.obs[obs_columns].dropna()

    # Compute correlation matrix
    corr_matrix = df.corr(method=method)

    # Mask upper triangle to remove redundant half
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'{method.capitalize()} Correlation Heatmap')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return plt.gcf()

def get_lsc_blast_genes(adata, topgenes_df, usage_norm, wdir, sample_name):
    """
    Select genes from LSC and blast-associated usages, including additional blast-like programs
    """
    signature_columns = [
        "LSC104_Ng2016_UP",
        "LSPC_Primed_Top100",
        "LSPC_Quiescent",
        "EPPERT_LSC_R",
        "EPPERT_CE_HSC_LSC",
        "GAL_LEUKEMIC_STEM_CELL_UP",
        "Blast_Score"
    ]

    # Get available signatures
    available_signatures = [col for col in signature_columns if col in adata.obs.columns]
    if not available_signatures:
        raise ValueError("No signature columns found in data")

    # Calculate correlations
    usage_cols = [col for col in adata.obs.columns if col.startswith('Usage_')]
    corr_matrix = adata.obs[usage_cols + available_signatures].corr(method='spearman')
    usage_sig_corr = corr_matrix.loc[usage_cols, available_signatures]

    print("\nCorrelations between usages and signatures:")
    print(usage_sig_corr)

    # Find LSC usage (highest average correlation with LSC signatures)
    lsc_signatures = [sig for sig in available_signatures if sig != "Blast_Score"]
    lsc_corrs = usage_sig_corr[lsc_signatures].mean(axis=1)
    lsc_usage = lsc_corrs.idxmax()

    # Find blast usages
    blast_usages = []
    if "Blast_Score" in available_signatures:
        blast_score_corr = usage_sig_corr["Blast_Score"]

        # Get correlations with LSC usage
        usage_usage_corr = corr_matrix.loc[usage_cols, usage_cols]
        lsc_neg_corr = usage_usage_corr[lsc_usage]

        # Set correlation threshold
        BLAST_CORR_THRESHOLD = 0.15

        # Find all programs with high blast correlation (excluding LSC program)
        for usage in usage_cols:
            if usage != lsc_usage:
                blast_corr = blast_score_corr[usage]
                if blast_corr > BLAST_CORR_THRESHOLD:
                    blast_usages.append(usage)

        print("\nBlast usage selection scores:")
        print("Usage\tBlast_Score_Corr\tLSC_Anti_Corr\tSelected")
        for usage in usage_cols:
            if usage != lsc_usage:
                blast_corr = blast_score_corr[usage]
                lsc_anti_corr = -lsc_neg_corr[usage]  # Keep this for information only
                is_selected = usage in blast_usages
                print(f"{usage}\t{blast_corr:.3f}\t{lsc_anti_corr:.3f}\t{'Yes' if is_selected else 'No'}")

    if not blast_usages:
        print("\nWarning: No programs met all criteria. Selecting program with highest blast correlation.")
        blast_usages = [blast_score_corr.drop(lsc_usage).idxmax()]

    print(f"\nIdentified LSC usage: {lsc_usage}")
    print(f"Identified blast usages: {blast_usages}")

    # Get genes from all identified programs
    lsc_genes = list(topgenes_df[lsc_usage].dropna().values)

    # Create a dictionary to store genes from each blast usage
    blast_usage_genes = {}
    for blast_usage in blast_usages:
        blast_usage_genes[blast_usage] = list(topgenes_df[blast_usage].dropna().values)

    # Create DataFrame with LSC and individual blast usage genes
    max_len = max(
        len(lsc_genes),
        max((len(genes) for genes in blast_usage_genes.values()), default=0)
    )

    # Create the base dictionary with LSC genes
    gene_dict = {
        'LSC_usage': lsc_genes + [''] * (max_len - len(lsc_genes))
    }

    # Add individual blast usage columns with incrementing numbers
    for i, (usage, genes) in enumerate(blast_usage_genes.items(), 1):
        gene_dict[f'Blast_Usage_{i}'] = genes + [''] * (max_len - len(genes))

    # Create DataFrame
    gene_df = pd.DataFrame(gene_dict)

    # Save the DataFrame
    genes_file = os.path.join(wdir, sample_name, "NMF_K10", "results",
                             f"{sample_name}_selected_genes.csv")
    gene_df.to_csv(genes_file, index=False)
    print(f"Saved LSC and blast genes to {genes_file}")

    # Return all genes for compatibility
    all_genes = []
    all_genes.extend(lsc_genes)
    for genes in blast_usage_genes.values():
        all_genes.extend(genes)
    return list(dict.fromkeys(all_genes))  # Return unique genes

def save_usage_umaps(adata, K, sample_name, wdir):
    """
    Create and save UMAP plots colored by usage scores
    """
    # Create UMAP plots directory
    umap_dir = os.path.join(wdir, sample_name, "NMF_K10", "results", "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Generate list of usage columns
    usage_columns = [f"Usage_{i+1}" for i in range(K)]

    # Create and save plot
    umap_file = os.path.join(umap_dir, f"{sample_name}_usage_umaps.png")

    # Create figure
    plt.figure(figsize=(15, 10))
    sc.pl.embedding(
        adata,
        color=usage_columns,
        basis="X_umap",
        legend_fontsize=5,
        legend_fontweight="medium",
        legend_fontoutline=3,
        ncols=3,
        color_map="RdGy_r",
        palette="tab20",
        show=False,
        save=False  # Don't let scanpy save the file
    )

    # Save using matplotlib
    plt.savefig(umap_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved usage UMAP plots to {umap_file}")

def get_malignant_cells(adata, sample_name):
    """
    Get malignant cells based on malignancy scores and status
    """
    # Print paths for debugging
    print(f"Sample name: {sample_name}")
    print(f"Sample path: {adata.uns['sample_path']}")

    malignancy_file = os.path.join(adata.uns['sample_path'],
                                  'malignant_normal_new',
                                  f'{sample_name}_cell_malignancy_scores.csv')

    print(f"Looking for malignancy file at: {malignancy_file}")

    if not os.path.exists(malignancy_file):
        raise FileNotFoundError(f"Malignancy scores file not found for {sample_name}")

    # Read malignancy data
    malignancy_df = pd.read_csv(malignancy_file, index_col=0)

    # Add malignancy information to adata
    adata.obs['malignancy_score'] = malignancy_df['malignancy_score']
    adata.obs['malignancy_status'] = malignancy_df['malignancy_status']

    # Filter cells based on malignancy criteria
    malignant_mask = (
        (adata.obs['malignancy_status'] == 'malignant') &
        (adata.obs['malignancy_score'] > 0.9)
    )

    print(f"Found {malignant_mask.sum()} malignant cells out of {len(adata)}")

    return adata[malignant_mask].copy()

def subsample_cells(adata, max_cells=4000, seed=42):
    """
    Subsample cells if the dataset is larger than max_cells

    Args:
        adata: AnnData object
        max_cells: Maximum number of cells to keep
        seed: Random seed for reproducibility

    Returns:
        AnnData: Subsampled or original AnnData object
    """
    if adata.n_obs > max_cells:
        print(f"\nDataset contains {adata.n_obs} cells, subsampling to {max_cells} cells")
        np.random.seed(seed)
        cells_idx = np.random.choice(adata.n_obs, size=max_cells, replace=False)
        return adata[cells_idx].copy()
    return adata

def process_sample(sample_path, output_dir=None):
    """
    Process a single sample
    """
    sample_name = os.path.basename(sample_path)
    print(f"\nProcessing sample: {sample_name}")

    # Read the muon file
    mdata = mu.read(os.path.join(sample_path, f"{sample_name}.h5mu"))
    print("read muon file")

    # Store sample path for later use
    mdata['rna'].uns['sample_path'] = sample_path

    # Get malignant cells using new criteria
    mdata['rna'] = get_malignant_cells(mdata['rna'], sample_name)

    # Rest of the processing...

def process_muon_files(directory_path, K, numiter, numhvgenes, wdir, specific_samples=None):
    """
    Process all muon files in the specified directory and perform NMF

    Args:
        directory_path (str): Path to the directory containing sample folders
        K: Number of components for NMF
        numiter: Number of iterations for NMF
        numhvgenes: Number of highly variable genes
        wdir: Working directory path
        specific_samples (list, optional): List of specific samples to process
    """
    data_dir = Path(directory_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Read meta annotation file
    meta_annotation_file = "/meta_annotation.csv"
    meta_annotation = pd.read_csv(meta_annotation_file, index_col=0)

    results = {}
    for sample_dir in data_dir.iterdir():
        if sample_dir.is_dir():
            sample_name = sample_dir.name

            # Skip if not in the specified samples
            if specific_samples and sample_name not in specific_samples:
                continue

            muon_file = sample_dir / f"{sample_name}.h5mu"

            if not muon_file.exists():
                print(f"Warning: No h5mu file found for sample {sample_name}")
                continue

            try:
                print(f"Processing sample: {sample_name}")

                # Read muon file
                mdata = mu.read(muon_file)
                mdata['rna'].uns['sample_path'] = str(sample_dir)

                # Add meta annotation
                print("Adding meta annotation...")
                mdata['rna'].obs['annotation'] = meta_annotation["annotation"]

                # Filter out NK, B, and T cells
                cells_to_remove = mdata['rna'].obs['annotation'].isin(['B Cell', "NK Cells", "T Cell"])
                if cells_to_remove.any():
                    print(f"Removing {cells_to_remove.sum()} NK, B, and T cells")
                    mdata_sub = mdata['rna'][~cells_to_remove].copy()
                else:
                    mdata_sub = mdata['rna'].copy()

                # Remove ribosomal genes and subset to RNA modality
                features_to_remove = [i for i in mdata_sub.var_names
                                    if i.startswith("RPS") or i.startswith("RPL")]
                mdata_sub = mdata_sub[:, ~mdata_sub.var_names.isin(features_to_remove)].copy()

                print(f"Removed {len(features_to_remove)} ribosomal genes")
                print(f"Initial features: {mdata_sub.shape[1]}")
                print(f"Initial cells: {mdata_sub.shape[0]}")

                # Get malignant cells using new criteria
                mdata_sub = get_malignant_cells(mdata_sub, sample_name)

                # Add subsampling step here
                mdata_sub = subsample_cells(mdata_sub, max_cells=8000)

                print(f"Final dimensions: {mdata_sub.shape[0]} cells x {mdata_sub.shape[1]} features")

                # Perform NMF and get results
                topgenes, usage_norm = do_NMF(
                    mdata_sub,
                    K,
                    numiter,
                    numhvgenes,
                    wdir,
                    sample_name
                )

                # Create UMAP plots
                save_usage_umaps(mdata_sub, K[0], sample_name, wdir)

                # Create correlation scatter plots
                create_correlation_scatter_plots(mdata_sub, sample_name, wdir)

                # Store results
                results[sample_name] = {
                    'topgenes': topgenes,
                    'usage_norm': usage_norm
                }

                # Get LSC and blast genes
                selected_genes = get_lsc_blast_genes(
                    mdata_sub,
                    topgenes,
                    usage_norm,
                    wdir,
                    sample_name
                )
                print(f"Selected {len(selected_genes)} genes from LSC and blast usages")

                # Create gene correlation plot using selected genes
                gene_plot_file = os.path.join(wdir, sample_name, "NMF_K10", "results",
                                            f"{sample_name}_lsc_blast_gene_correlation.png")
                plot_gene_correlation(mdata_sub, selected_genes,
                                   layer='lognorm_counts',
                                   method='spearman',
                                   output_file=gene_plot_file)
                print(f"Created LSC/blast gene correlation plot for {sample_name}")

                # Create usage and signature correlation plot
                usage_columns = [f"Usage_{i+1}" for i in range(K[0])]
                signature_columns = [
                    "LSC104_Ng2016_UP",
                    "LSPC_Primed_Top100",
                    "LSPC_Quiescent",
                    "EPPERT_LSC_R",
                    "EPPERT_CE_HSC_LSC",
                    "GAL_LEUKEMIC_STEM_CELL_UP",
                    "Blast_Score"
                ]

                # Check which signature columns are actually present
                available_signatures = [col for col in signature_columns if col in mdata_sub.obs.columns]
                if available_signatures:
                    plot_columns = usage_columns + available_signatures
                    usage_plot_file = os.path.join(wdir, sample_name, "NMF_K10", "results",
                                                 f"{sample_name}_usage_signature_correlation.png")
                    correlation_heatmap(mdata_sub, plot_columns,
                                     method='spearman',
                                     output_file=usage_plot_file)
                    print(f"Created usage/signature correlation plot for {sample_name}")
                else:
                    print(f"Warning: No signature columns found in data for {sample_name}")

                # Create LSC-Blast-Usage space plot
                create_lsc_blast_usage_space(mdata_sub, sample_name, wdir)

                # Calculate and plot program similarities
                similarity_dict = calculate_program_similarities(topgenes, usage_norm, sample_name, wdir)

            except Exception as e:
                print(f"Error processing {sample_name}: {str(e)}")
                continue

    # After processing all samples, create correlation plots
    if len(results) > 0:
        # For each sample, create correlation plots
        for sample_name, sample_results in results.items():
            try:
                # Read the saved adata file with usage scores
                adata_file = os.path.join(wdir, sample_name, "NMF_K10", "results",
                                        f"{sample_name}_with_usage.h5ad")
                adata = sc.read(adata_file)

                # Get LSC and blast genes
                selected_genes = get_lsc_blast_genes(
                    adata,
                    sample_results['topgenes'],
                    sample_results['usage_norm'],
                    wdir,
                    sample_name
                )
                print(f"Selected {len(selected_genes)} genes from LSC and blast usages")

                # Create gene correlation plot using selected genes
                gene_plot_file = os.path.join(wdir, sample_name, "NMF_K10", "results",
                                            f"{sample_name}_lsc_blast_gene_correlation.png")
                plot_gene_correlation(adata, selected_genes,
                                   layer='lognorm_counts',
                                   method='spearman',
                                   output_file=gene_plot_file)
                print(f"Created LSC/blast gene correlation plot for {sample_name}")

            except Exception as e:
                print(f"Error creating plots for {sample_name}: {str(e)}")

    return results

def create_correlation_scatter_plots(adata, sample_name, wdir):
    """
    Create scatter plots comparing usages with LSC and blast scores
    """
    print("\nCreating correlation scatter plots...")

    # Create correlation plots directory
    corr_dir = os.path.join(wdir, sample_name, "NMF_K10", "results", "correlation_scatter")
    os.makedirs(corr_dir, exist_ok=True)

    # Get usages
    usage_cols = [col for col in adata.obs.columns if col.startswith('Usage_')]
    if len(usage_cols) < 2:
        print("Warning: Less than 2 usage components found")
        return

    # Get scores
    score_cols = ["LSC104_Ng2016_UP", "Blast_Score", "EPPERT_LSC_R", "EPPERT_CE_HSC_LSC",
                 "GAL_LEUKEMIC_STEM_CELL_UP", "LSPC_Primed_Top100", "LSPC_Quiescent"]
    available_scores = [col for col in score_cols if col in adata.obs.columns]
    if not available_scores:
        print("Warning: No LSC or Blast scores found")
        return

    # Create all combinations of plots
    plot_pairs = []

    # Usage1 vs Usage2
    plot_pairs.append((usage_cols[0], usage_cols[1], 'Usage_1', 'Usage_2'))

    # Usages vs Scores
    for usage in usage_cols:
        for score in available_scores:
            plot_pairs.append((usage, score, usage, score))

    # Create plots
    for x_col, y_col, xlabel, ylabel in plot_pairs:
        plt.figure(figsize=(8, 8))

        # Get data
        x = adata.obs[x_col]
        y = adata.obs[y_col]

        # Create scatter plot
        plt.scatter(x, y, alpha=0.5, s=1, color='blue')

        # Calculate correlation
        corr = adata.obs[[x_col, y_col]].corr(method='spearman').iloc[0, 1]

        # Fit line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_line, p(x_line), 'r--', alpha=0.8)

        # Add equation of line
        #equation = f'y = {z[0]:.2f}x + {z[1]:.2f}'
        plt.title(f'Spearman Correlation: {corr:.3f}')

        # Add labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Save plot
        plt.savefig(
            os.path.join(corr_dir, f"{sample_name}_{xlabel}_vs_{ylabel}_scatter.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    print(f"Saved correlation scatter plots to {corr_dir}")

def create_lsc_blast_usage_space(adata, sample_name, wdir):
    """
    Create a scatter plot of Usage_1 vs Usage_2 colored by LSC-Blast difference
    """
    # Create plot directory
    plot_dir = os.path.join(wdir, sample_name, "NMF_K10", "results", "correlation_scatter")
    os.makedirs(plot_dir, exist_ok=True)

    if not all(col in adata.obs.columns for col in ['Usage_1', 'Usage_2', 'LSPC_Primed_Top100', 'Blast_Score']):
        print("Warning: Missing required scores")
        return

    # Create figure
    plt.figure(figsize=(10, 10))

    # Create scatter plot
    scatter = plt.scatter(
        adata.obs['Usage_1'],
        adata.obs['Usage_2'],
        c=adata.obs['LSPC_Primed_Top100'] - adata.obs['Blast_Score'],  # Color by LSC-Blast difference
        cmap='RdBu_r',  # Red for high LSC, Blue for high Blast
        s=5,
        alpha=0.6
    )

    # Add colorbar
    plt.colorbar(scatter, label='LSC - Blast Score')

    # Calculate correlations
    corr_u1_lsc = adata.obs[['Usage_1', 'LSPC_Primed_Top100']].corr(method='spearman').iloc[0, 1]
    corr_u1_blast = adata.obs[['Usage_1', 'Blast_Score']].corr(method='spearman').iloc[0, 1]
    corr_u2_lsc = adata.obs[['Usage_2', 'LSPC_Primed_Top100']].corr(method='spearman').iloc[0, 1]
    corr_u2_blast = adata.obs[['Usage_2', 'Blast_Score']].corr(method='spearman').iloc[0, 1]

    # Add title and labels
    plt.title(f'LSC-Blast-Usage\n' +
             f'U1-LSC: {corr_u1_lsc:.3f}, U1-Blast: {corr_u1_blast:.3f}\n' +
             f'U2-LSC: {corr_u2_lsc:.3f}, U2-Blast: {corr_u2_blast:.3f}')
    plt.xlabel('Usage 1')
    plt.ylabel('Usage 2')

    # Save plot
    plt.savefig(
        os.path.join(plot_dir, f"{sample_name}_lsc_blast_usage_space.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print(f"Saved LSC-Blast-Usage space plot")

def calculate_program_similarities(topgenes_df, usage_norm, sample_name, results_directory):
    """
    Calculate and plot both Jaccard and cosine similarities between programs
    """
    n_programs = len(usage_norm.columns)

    # Initialize similarity matrices
    jaccard_matrix = np.zeros((n_programs, n_programs))
    cosine_matrix = cosine_similarity(usage_norm.T)

    # Get program columns from topgenes DataFrame
    program_cols = [col for col in topgenes_df.columns if col.startswith('Usage_')]

    print(f"Found program columns: {program_cols}")  # Debug print

    # Calculate Jaccard similarities
    for i in range(n_programs):
        for j in range(n_programs):
            # Get column names for both programs
            col1 = f'Usage_{i+1}'
            col2 = f'Usage_{j+1}'

            try:
                # Get gene sets, dropping NaN values
                genes1 = set(topgenes_df[col1].dropna().values)
                genes2 = set(topgenes_df[col2].dropna().values)

                # Calculate Jaccard similarity
                intersection = len(genes1.intersection(genes2))
                union = len(genes1.union(genes2))
                jaccard_matrix[i,j] = intersection / union if union > 0 else 0

            except KeyError as e:
                print(f"Warning: Could not find column {e} in topgenes DataFrame")
                print(f"Available columns: {topgenes_df.columns.tolist()}")
                jaccard_matrix[i,j] = 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Jaccard similarities
    sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax1,
        square=True,
        xticklabels=[f'Usage_{i+1}' for i in range(n_programs)],
        yticklabels=[f'Usage_{i+1}' for i in range(n_programs)]
    )
    ax1.set_title(f'{sample_name} Program Jaccard Similarities')

    # Plot Cosine similarities
    sns.heatmap(
        cosine_matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        ax=ax2,
        square=True,
        xticklabels=[f'Usage_{i+1}' for i in range(n_programs)],
        yticklabels=[f'Usage_{i+1}' for i in range(n_programs)]
    )
    ax2.set_title(f'{sample_name} Program Cosine Similarities')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_directory, f"{sample_name}_program_similarities.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save similarity matrices
    similarity_dict = {
        'jaccard': pd.DataFrame(
            jaccard_matrix,
            columns=[f'Usage_{i+1}' for i in range(n_programs)],
            index=[f'Usage_{i+1}' for i in range(n_programs)]
        ),
        'cosine': pd.DataFrame(
            cosine_matrix,
            columns=[f'Usage_{i+1}' for i in range(n_programs)],
            index=[f'Usage_{i+1}' for i in range(n_programs)]
        )
    }

    # Save matrices to CSV
    for sim_type, matrix in similarity_dict.items():
        matrix_path = os.path.join(results_directory, f"{sample_name}_{sim_type}_similarities.csv")
        matrix.to_csv(matrix_path)

    print(f"Saved similarity plots and matrices to {results_directory}")

    return similarity_dict

def plot_program_scores(adata, sample_name, results_directory):
    """
    Create a plot showing the distribution and mean scores for each program across all cells
    """
    # Get usage columns
    usage_cols = [col for col in adata.obs.columns if col.startswith('Usage_')]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create violin plot
    sns.violinplot(data=adata.obs[usage_cols], cut=0)

    # Add mean scores
    means = adata.obs[usage_cols].mean()
    plt.plot(range(len(usage_cols)), means, 'ro', label='Mean')

    # Add mean values as text
    for i, mean in enumerate(means):
        plt.text(i, mean, f'{mean:.2f}', ha='center', va='bottom')

    # Customize plot
    plt.title(f'{sample_name} Program Usage Scores')
    plt.xlabel('Programs')
    plt.ylabel('Usage Score')
    plt.xticks(range(len(usage_cols)), usage_cols, rotation=45)
    plt.legend()

    # Save plot
    plot_path = os.path.join(results_directory, f"{sample_name}_program_scores.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved program scores plot to {results_directory}")

# Example usage
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process muon files with NMF')
    parser.add_argument('data_directory', type=str,
                       help='Directory containing the muon files')
    parser.add_argument('--sample', type=str, nargs='+',
                       help='Specific sample(s) to process (optional). Can provide multiple samples')
    parser.add_argument('--working-dir', type=str,
                       default="/samples/",
                       help='Working directory for output')
    parser.add_argument('--k', type=int, default=6,
                       help='Number of components for NMF')
    parser.add_argument('--numiter', type=int, default=50,
                       help='Number of iterations')
    parser.add_argument('--numhvgenes', type=int, default=2000,
                       help='Number of highly variable genes')

    # Parse arguments
    args = parser.parse_args()

    # Run the processing
    results = process_muon_files(
        args.data_directory,
        [args.k],  # Wrap in list as required by cNMF
        args.numiter,
        args.numhvgenes,
        args.working_dir,
        args.sample
    )


#python process_muons.py /sample \
#    --sample AML3 AML4 AML5 AML6 AML7 AML8 \
#          AML9_Dx AML9_Rel AML10_Dx AML10_Rel \
#          AML11_Dx AML11_Rel AML12_Dx AML12_Rel \
#          AML13_Dx AML13_Rel AML14_Dx AML14_Rel \
#          AML15_Dx AML15_Rel AML16_Rel AML17_Rel \
#          AML19_Dx AML19_Rel
