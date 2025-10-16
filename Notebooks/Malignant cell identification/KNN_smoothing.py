import os
import muon as mu
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
import json
from scipy.stats import gaussian_kde
import decoupler as dc


sc._settings.settings._vector_friendly=True

# Define color dictionaries for different features
CELLTYPE_COLORS = {
    'Erythroid_Prog': '#E41A1C',  # red
    'Progenitors': '#377EB8',     # blue
    'HSC': '#984EA3',            # changed to purple
    'Erythroid': '#66C2A5',      # changed to mint
    'GMP': '#FF7F00',           # orange
    'B_prog': '#FFFF33',        # yellow
    'B Cell': '#A65628',        # brown
    'T Cells': '#F781BF',       # pink
    'Monocytes': '#E6AB02',     # changed to gold
    'NK Cells': '#A6D854',      # changed to lime
    'ProMono': '#FC8D62',       # coral
    'DCs': '#8DA0CB',           # light blue
    "Cycling" : "green"
}

TYPE_COLORS = {
    'Healthy': '#ADD8E6',  # lightblue
    'Tumor': '#A65628'     # brown
}

MALIGNANCY_COLORS = {
    'normal': '#377EB8',   # blue
    'malignant': '#E41A1C' # red
}

# Define fusion cleaner function
def fus_cleaner(fusion):
    """Clean and standardize fusion names"""
    if fusion in ["CBFB--MYH11"]:
        return "CBFB--MYH11"
    if fusion in ["KMT2A--AFDN"]:
        return "KMT2A--AFDN"
    if fusion in ["NUP214--SET"]:
        return "NUP214--SET"
    if fusion in ["KMT2A--MLLT10"]:
        return "KMT2A--MLLT10"
    if fusion in ["RUNX1--RUNX1T1"]:
        return "RUNX1--RUNX1T1"
    if fusion in ["DEK--NUP214"]:
        return "DEK--NUP214"
    if fusion in ["NUP98--NSD1"]:
        return "NUP98--NSD1"
    if fusion in ["CBFA2T3--GLIS2"]:
        return "CBFA2T3--GLIS2"
    else:
        return None

def load_fusion_data():
    """Load and process fusion data"""
    print("Loading fusion data...")

    # Load fusion data from JSON
    with open("/pAML_fusions.json") as f:
        fus_data = json.load(f)

    barcode_fusion_list = []

    for sample, sample_data in fus_data.items():
        clean_sample = str(sample).replace("_R2", "")  # remove _R2
        fins = sample_data.get("fins", {})
        barcodes = fins.get("barcodes", {})
        coding = barcodes.get("coding", {})

        for fusion_key, fusion_entries in coding.items():
            for entry in fusion_entries:
                barcode_fusion_list.append({
                    "sample": clean_sample,
                    "barcode": f"{entry['barcode']}-1_{clean_sample}",  # add -1 and sample
                    "fusion_name": entry["FusionName"],
                    "fusion_id": entry["Fusion"]
                })

    barcode_fusion_df = pd.DataFrame(barcode_fusion_list)

    barcode_fusion_df.set_index("barcode",inplace=True)
    fus_meta = barcode_fusion_df[~barcode_fusion_df.index.duplicated(keep='first')]


    # Clean fusion names
    fus_meta["fusion_clean"] = fus_meta["fusion_name"].apply(fus_cleaner)

    # Remove None values
    fus_meta = fus_meta.dropna(subset=["fusion_clean"])

    print(f"Loaded fusion data for {len(fus_meta)} cells with {fus_meta['fusion_clean'].nunique()} unique fusions")
    return fus_meta

def sc_wf_alclus(adata, res=2, HVG=800):
    """
    Workflow for clustering analysis including harmony integration
    """

    adata.X = adata.layers["raw_counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=HVG,batch_key = "sample", span=0.3)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')


    sc.pp.neighbors(
        adata,
        n_neighbors=100,
        n_pcs=30,
        use_rep="X_pca"
    )
    #sc.external.pp.bbknn(adata, "sample")
    sc.tl.umap(
        adata,
        min_dist=0.3,
        spread=1.5
    )
    adata.obsm["uninteg_UMAP"] = adata.obsm["X_umap"]
    sc.tl.leiden(
        adata,
        resolution=res,
        key_added="leiden_uninteg",
        use_weights=False
    )

    return adata

def add_malignancy_score(adata, sample_type_col='type', leiden_col='leiden_uninteg'):
    """
    Calculate malignancy scores for each leiden cluster
    """
    df = adata.obs[[sample_type_col, leiden_col]].copy()
    malignancy_scores = (
        df.groupby(leiden_col)[sample_type_col]
        .apply(lambda x: (x == 'Tumor').sum() / len(x))
    )
    return malignancy_scores

def refine_malignancy_status(adata, n_neighbors=50, threshold=0.2, max_iterations=20):
    """
    Iteratively refine malignancy status using KNN voting until convergence
    """
    print("Refining malignancy status iteratively...")

    # Create initial malignancy_status from the "type" column
    adata.obs["malignancy_status"] = adata.obs["type"].map({"Healthy": "normal", "Tumor": "malignant"})

    # Get KNN graph
    knn_graph = adata.obsp["distances"]

    # Initialize arrays for storing results
    healthy_neighbor_counts = np.zeros(adata.n_obs)
    healthy_neighbor_proportions = np.zeros(adata.n_obs)

    # Track changes and status counts over iterations
    iteration_changes = []
    normal_counts = []
    malignant_counts = []

    iteration = 0
    changes = float('inf')

    while changes > 0 and iteration < max_iterations:
        changes = 0
        previous_status = adata.obs["malignancy_status"].copy()

        # Reassign labels based on neighbors
        for i in range(adata.n_obs):
            # Get indices of the nearest neighbors
            neighbors = knn_graph[i].indices[:50]

            # Get neighbor labels for healthy proportion calculation
            neighbor_types = adata.obs.iloc[neighbors]["type"]

            # Count healthy cells in neighborhood
            healthy_count = (neighbor_types == "Healthy").sum()
            healthy_neighbor_counts[i] = healthy_count
            healthy_neighbor_proportions[i] = healthy_count / len(neighbors)

            # Use voting neighbors for malignancy refinement
            voting_neighbors = neighbors[:n_neighbors]
            voting_labels = adata.obs.iloc[voting_neighbors]["malignancy_status"]

            # Count votes for malignancy status
            normal_count = (voting_labels == "normal").sum()
            malignant_count = (voting_labels == "malignant").sum()
            total = normal_count + malignant_count

            # Apply threshold for relabeling
            if total > 0:
                normal_ratio = normal_count / total
                malignant_ratio = malignant_count / total

                current_status = adata.obs.at[adata.obs.index[i], "malignancy_status"]
                new_status = current_status

                if normal_ratio >= threshold:
                    new_status = "normal"
                elif malignant_ratio >= threshold:
                    new_status = "malignant"

                if new_status != current_status:
                    adata.obs.at[adata.obs.index[i], "malignancy_status"] = new_status
                    changes += 1

        # Track statistics
        iteration_changes.append(changes)
        normal_counts.append((adata.obs["malignancy_status"] == "normal").sum())
        malignant_counts.append((adata.obs["malignancy_status"] == "malignant").sum())

        iteration += 1
        print(f"Iteration {iteration}: {changes} cells changed status")

    # Create convergence plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(range(1, iteration + 1), iteration_changes, 'b-', marker='o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Changes')
    ax1.set_title('Convergence Pattern')
    ax1.grid(True)

    ax2.plot(range(1, iteration + 1), normal_counts, 'b-', label='Normal', marker='o')
    ax2.plot(range(1, iteration + 1), malignant_counts, 'r-', label='Malignant', marker='o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cell Count')
    ax2.set_title('Cell Type Distribution')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save
    convergence_plot_path = os.path.join(
        os.path.dirname(adata.uns.get('output_dir', '.')),
        'convergence_plot.pdf'
    )
    plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add convergence statistics to adata
    adata.uns['refinement_stats'] = {
        'iterations': iteration,
        'changes_per_iteration': iteration_changes,
        'normal_counts': normal_counts,
        'malignant_counts': malignant_counts
    }

    # Add healthy neighbor metrics to adata
    adata.obs['healthy_neighbor_count'] = healthy_neighbor_counts
    adata.obs['healthy_neighbor_proportion'] = healthy_neighbor_proportions

    print(f"Malignancy status refinement complete after {iteration} iterations")
    print(f"Added healthy_neighbor_count and healthy_neighbor_proportion to adata.obs")

    # Calculate and print statistics
    final_normal = (adata.obs["malignancy_status"] == "normal").sum()
    final_malignant = (adata.obs["malignancy_status"] == "malignant").sum()
    print(f"Final statistics:")
    print(f"Normal cells: {final_normal}")
    print(f"Malignant cells: {final_malignant}")
    print(f"Normal percentage: {(final_normal/adata.n_obs)*100:.2f}%")

    return adata

def save_umaps(adata, output_dir, sample_name):
    """
    Save individual UMAP plots for different colorings and malignancy analysis plots
    """
    # Create UMAP directory
    umap_dir = os.path.join(output_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)


    # Load fusion data
    try:
        fus_meta = load_fusion_data()

        # Add fusion information to adata
        adata.obs["fusion_clean"] = None

        # Match fusion data with adata cells
        common_cells = adata.obs.index.intersection(fus_meta.index)
        if len(common_cells) > 0:


            adata.obs.loc[common_cells, "fusion_clean"] = fus_meta.loc[common_cells, "fusion_clean"]
            print(f"Added fusion data for {len(common_cells)} cells")
        else:
            print("No matching cells found between adata and fusion data")

    except Exception as e:
        print(f"Warning: Could not load fusion data: {e}")
        fus_meta = None

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(
        adata.obs['malignancy_score'],
        adata.obs['healthy_neighbor_proportion'],
        alpha=0.5,
        s=1,
        c=adata.obs['malignancy_score'],
        cmap='coolwarm'
    )

    # Add labels and title
    plt.xlabel('Malignancy Score')
    plt.ylabel('Proportion of Healthy Neighbors')
    plt.title(f"{sample_name} - Malignancy Score vs Healthy Neighbors\n"
              "High malignancy scores correlate with fewer healthy neighbors")

    # Add colorbar
    plt.colorbar(label='Malignancy Score')

    # Add horizontal lines to show thresholds
    plt.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.02, 0.21, 'Healthy threshold (20%)', fontsize=8)

    # Save plot
    plt.savefig(
        os.path.join(umap_dir, f"{sample_name}_malignancy_analysis.pdf"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save regular UMAP plots
    features = ["type", "leiden_uninteg", "malignancy_status", "malignancy_score",
               "annotation", "healthy_neighbor_count", "healthy_neighbor_proportion"]

    for feature in features:
        plt.figure(figsize=(8, 8))
        if feature == "malignancy_score" or "neighbor" in feature:
            sc.pl.umap(adata, color=feature, cmap="coolwarm", show=False)
        elif feature == "Blast_Score":
            # Calculate percentiles for Blast_Score
            blast_scores = adata.obs[feature].dropna()
            if len(blast_scores) > 0:
                vmin = np.percentile(blast_scores, 5)
                vmax = np.percentile(blast_scores, 95)
                sc.pl.umap(adata, color=feature, cmap="coolwarm", vmin=vmin, vmax=vmax, show=False)
            else:
                sc.pl.umap(adata, color=feature, cmap="coolwarm", show=False)
        elif feature == "annotation":
            sc.pl.umap(adata, color=feature, palette=CELLTYPE_COLORS, show=False)
        elif feature == "type":
            sc.pl.umap(adata, color=feature, palette=TYPE_COLORS, show=False)
        elif feature == "malignancy_status":
            sc.pl.umap(adata, color=feature, palette=MALIGNANCY_COLORS, show=False)
        else:
            sc.pl.umap(adata, color=feature, show=False)

        plt.title(f"{sample_name} - {feature}")
        plt.savefig(
            os.path.join(umap_dir, f"{sample_name}_umap_{feature}.pdf"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # Add fusion-specific UMAP plots if fusion data is available
    if fus_meta is not None and 'fusion_clean' in adata.obs.columns:
        # Get fusion statistics to find the most common fusion
        fusion_cells = adata.obs['fusion_clean'].notna()
        if fusion_cells.sum() > 0:
            fusion_stats = adata.obs.loc[fusion_cells, 'fusion_clean'].value_counts()
            top_fusion = fusion_stats.index[0]  # Get the fusion with highest count
            print(f"Most common fusion: {top_fusion} with {fusion_stats.iloc[0]} cells")

            # Get UMAP coordinates
            umap_coords = adata.obsm['X_umap']

            # Create plot for top fusion only
            plt.figure(figsize=(10, 8))

            # Plot all cells in light gray
            plt.scatter(umap_coords[:, 0], umap_coords[:, 1],
                       c='gray', alpha=0.3, s=1, label='Other cells', zorder=-1, rasterized=True)

            # Get cells with top fusion
            top_fusion_mask = adata.obs['fusion_clean'] == top_fusion
            top_fusion_coords = umap_coords[top_fusion_mask]

            if len(top_fusion_coords) > 10:  # Need enough points for KDE
                # Create KDE for top fusion
                kde = gaussian_kde(top_fusion_coords.T)

                # Create grid for KDE evaluation
                x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
                y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()

                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])

                # Evaluate KDE
                density = kde(positions).reshape(xx.shape)

                # Calculate density threshold to mask low density areas
                # Use 5% of maximum density as threshold to mask low density
                density_threshold = density.max() * 0.05
                density_masked = np.where(density > density_threshold, density, np.nan)

                # Plot KDE contour with masking
                contour = plt.contourf(xx, yy, density_masked, levels=20, alpha=0.6, cmap='Reds')

                # Plot top fusion cells
                plt.scatter(top_fusion_coords[:, 0], top_fusion_coords[:, 1], rasterized=True,zorder=-1,
                           c='red', alpha=0.8, s=3, edgecolors='black', linewidth=0.5,
                           label=f'{top_fusion} ({len(top_fusion_coords)} cells)')

                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')
                plt.title(f"{sample_name} - {top_fusion} (KDE Density, Low Density Masked)")
                plt.colorbar(contour, label='Density')
                plt.legend()

                plt.savefig(
                    os.path.join(umap_dir, f"{sample_name}_umap_{top_fusion.replace('--', '_')}_kde_masked.pdf"),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()

                # Also create a version with higher density threshold for more aggressive masking
                plt.figure(figsize=(10, 8))

                # Plot all cells in light gray
                plt.scatter(umap_coords[:, 0], umap_coords[:, 1], rasterized=True,zorder=-1,
                           c='lightgray', alpha=0.3, s=1, label='Other cells')

                # Use 10% of maximum density for more aggressive masking
                density_threshold_high = density.max() * 0.10
                density_masked_high = np.where(density > density_threshold_high, density, np.nan)

                # Plot KDE contour with higher threshold masking
                contour_high = plt.contourf(xx, yy, density_masked_high, levels=20, alpha=0.6, cmap='Reds')

                # Plot top fusion cells
                plt.scatter(top_fusion_coords[:, 0], top_fusion_coords[:, 1], rasterized=True,zorder=-1,
                           c='red', alpha=0.8, s=3, edgecolors='black', linewidth=0.5,
                           label=f'{top_fusion} ({len(top_fusion_coords)} cells)')

                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')
                plt.title(f"{sample_name} - {top_fusion} (KDE Density, High Threshold Masking)")
                plt.colorbar(contour_high, label='Density')
                plt.legend()

                plt.savefig(
                    os.path.join(umap_dir, f"{sample_name}_umap_{top_fusion.replace('--', '_')}_kde_high_threshold.pdf"),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()

            # Create summary fusion statistics
            fusion_stats_file = os.path.join(umap_dir, f"{sample_name}_fusion_statistics.csv")
            fusion_stats.to_csv(fusion_stats_file)
            print(f"Saved fusion statistics to {fusion_stats_file}")
            print(f"Fusion statistics:\n{fusion_stats}")

    print(f"Saved UMAP and malignancy analysis plots to {umap_dir}")

def process_integration(tumor_sample):
    """
    Process tumor and healthy samples integration
    """
    print(f"Processing tumor sample: {tumor_sample}")

    # Create sample-specific output directory
    sample_output_dir = os.path.join("/Users/sina.kanannejad/Desktop/samplebsample",
                                   tumor_sample, "malignant_normal_new")
    os.makedirs(sample_output_dir, exist_ok=True)

    # Read tumor sample
    tumor_path = f"/sample/{tumor_sample}/{tumor_sample}.h5mu"
    mdata_tumor = mu.read(tumor_path)
    adata_tumor = mdata_tumor["rna"].copy()
    adata_tumor.X = adata_tumor.layers["raw_counts"].copy()
    adata_tumor.obs["type"] = "Tumor"

    # Clear mdata_tumor to free memory
    del mdata_tumor

    # Read healthy sample
    mdata_healthy = mu.read("/object_healthy.h5ad")
    mdata_healthy.obs["annotation"] = mdata_healthy.obs["cell_type_our_ref2"]
    mdata_healthy.obs["type"] = "Healthy"
    mdata_healthy.X = mdata_healthy.layers["raw_counts"].copy()

    # Read meta annotation
    meta_annotation = pd.read_csv("/meta_annotation.csv",
                                index_col=0)

    # Concatenate
    print("Concatenating samples...")
    integrated_obj = sc.concat([adata_tumor, mdata_healthy])

    # Add meta annotation
    print("Adding meta annotation...")
    integrated_obj.obs['annotation'] = meta_annotation["annotation"]

    # Print unique categories to debug
    print("Unique annotation categories:", integrated_obj.obs['annotation'].unique())

    # Clear individual objects to free memory
    del adata_tumor
    del mdata_healthy

    # Run workflow
    print("Running clustering workflow...")
    integrated_obj = sc_wf_alclus(integrated_obj)

    # Store output directory in adata
    integrated_obj.uns['output_dir'] = sample_output_dir

    # Refine malignancy status
    integrated_obj = refine_malignancy_status(integrated_obj)

    # Calculate malignancy scores
    print("Calculating malignancy scores...")
    scores = add_malignancy_score(integrated_obj, sample_type_col="type", leiden_col="leiden_uninteg")
    integrated_obj.obs['malignancy_score'] = integrated_obj.obs["leiden_uninteg"].map(scores).astype(float)

    # Save cluster-level scores
    scores_df = pd.DataFrame(scores).reset_index()
    scores_df.columns = ['Cluster', 'Malignancy_Score']
    cluster_scores_file = os.path.join(sample_output_dir, f"{tumor_sample}_cluster_malignancy_scores.csv")
    scores_df.to_csv(cluster_scores_file, index=False)
    print(f"Saved cluster malignancy scores to {cluster_scores_file}")

    # Save cell-level scores
    cell_scores_df = integrated_obj.obs[['leiden_uninteg', 'malignancy_score',
                                       'malignancy_status', 'annotation',
                                       'healthy_neighbor_count',
                                       'healthy_neighbor_proportion']].copy()
    cell_scores_file = os.path.join(sample_output_dir, f"{tumor_sample}_cell_malignancy_scores.csv")
    cell_scores_df.to_csv(cell_scores_file)
    print(f"Saved cell malignancy scores to {cell_scores_file}")

    # Save individual UMAP plots
    save_umaps(integrated_obj, sample_output_dir, tumor_sample)

    print(f"Processing complete for {tumor_sample}")
    return integrated_obj

def score_gene_lists(gene_sets: dict, adata, skip_sets=None, source_col="geneset", target_col="gene_symbol"):
    """
    Run AUCell scoring on an AnnData object for a dictionary of gene lists.

    Parameters:
    - gene_sets: dict
        Dictionary where keys are gene set names and values are lists of gene symbols.
    - adata: AnnData
        AnnData object containing the RNA data (e.g. adata["rna"]).
    - skip_sets: set or list, optional
        Names of gene sets to skip. Default is None.
    - source_col: str
        Name of the column representing gene set names in the DataFrame passed to AUCell.
    - target_col: str
        Name of the column representing gene symbols in the DataFrame passed to AUCell.
    """
    if skip_sets is None:
        skip_sets = set()

    for geneset_name, gene_list in gene_sets.items():
        if geneset_name in skip_sets:
            continue

        print(f"Scoring {geneset_name} ...")
        try:
            # Make DataFrame
            genes = list(set(gene_list))  # Deduplicate
            df = pd.DataFrame(genes, columns=[target_col])
            df[source_col] = geneset_name

            # Run AUCell
            dc.mt.aucell(
                adata,
                df,
                source=source_col,
                target=target_col,
                use_raw=False
            )
            # Store result in .obs
            adata.obs[geneset_name] = adata.obsm["aucell_estimate"][geneset_name]
            print(f"{geneset_name} is DONE!")

        except Exception as e:
            print(f"Error scoring {geneset_name}: {e}")

def calculate_blast_score(adata):
    """
    Calculate blast score using specific genes if Blast_Score doesn't exist
    """
    if 'Blast_Score' not in adata.obs.columns:
        print("Blast_Score not found, calculating from gene expression...")

        # Define blast score genes
        blast_genes = ["CLEC11A", "PRAME", "AZU1", "NREP", "ARMH1", "TRH", "C1QBP"]

        # Check which genes are available in the dataset
        available_genes = [gene for gene in blast_genes if gene in adata.var_names]
        missing_genes = [gene for gene in blast_genes if gene not in adata.var_names]

        if missing_genes:
            print(f"Warning: Missing genes for blast score calculation: {missing_genes}")

        if len(available_genes) > 0:
            print(f"Calculating blast score using {len(available_genes)} available genes: {available_genes}")

            # Create gene set dictionary
            gene_sets = {"Blast_Score": available_genes}

            # Calculate scores using AUCell
            try:
                score_gene_lists(gene_sets, adata)
                print("Blast score calculation completed successfully!")
            except Exception as e:
                print(f"Error calculating blast score: {e}")
                # Fallback: calculate mean expression
                print("Falling back to mean expression calculation...")
                adata.obs['Blast_Score'] = adata[:, available_genes].X.mean(axis=1)
        else:
            print("No blast score genes found in dataset. Creating dummy score.")
            adata.obs['Blast_Score'] = 0.0
    else:
        print("Blast_Score already exists in adata.obs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tumor samples with healthy integration')
    parser.add_argument('--sample', type=str, required=True,
                       help='Tumor sample to process (e.g., AML4)')

    args = parser.parse_args()

    # Process the sample
    integrated_obj = process_integration(args.sample)
