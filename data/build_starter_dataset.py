"""
脑靶向纳米粒子数据集构建脚本
从已发表论文中提取的实验数据
"""
import pandas as pd
import os

# ==============================================================
# 数据来源说明
# metric_type:
#   - pct_bbb_crossing: 体外BBB transwell过膜百分比 (%)
#   - relative_brain_counts: 体内脑内NP相对计数（以某组为1.0基准）
#   - pct_ID_brain: 注射剂量百分比/克脑组织 (%ID/g)
#   - pct_ID_brain_approx: %ID/g 近似估计
# ==============================================================

records = [

    # ──────────────────────────────────────────────────────────
    # Paper 1: Wiley et al. PNAS 2013
    # "Transcytosis and brain uptake of transferrin-containing
    #  nanoparticles by tuning avidity to transferrin receptor"
    # DOI: 10.1073/pnas.1307152110
    # NP type: gold, ligand: transferrin (Tf)
    # Outcome: relative brain parenchyma NP counts (image-based)
    # Avidity rule: intermediate Tf → optimal brain entry
    # ──────────────────────────────────────────────────────────
    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=21.0, zeta_mv=-12.9, peg="no",
         ligand="transferrin", ligand_count=3, kd_nM=4.9,
         brain_metric=0.5, metric_type="relative_brain_counts",
         model="mouse_iv", notes="5nm core, low avidity"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=25.3, zeta_mv=-10.0, peg="no",
         ligand="transferrin", ligand_count=6, kd_nM=3.1,
         brain_metric=0.6, metric_type="relative_brain_counts",
         model="mouse_iv", notes="5nm core"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=44.6, zeta_mv=-14.4, peg="no",
         ligand="transferrin", ligand_count=10, kd_nM=1.7,
         brain_metric=1.0, metric_type="relative_brain_counts",
         model="mouse_iv", notes="20nm core, reference group"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=43.6, zeta_mv=-6.8, peg="no",
         ligand="transferrin", ligand_count=20, kd_nM=1.5,
         brain_metric=1.1, metric_type="relative_brain_counts",
         model="mouse_iv", notes="20nm core"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=48.3, zeta_mv=-14.1, peg="no",
         ligand="transferrin", ligand_count=30, kd_nM=0.71,
         brain_metric=1.3, metric_type="relative_brain_counts",
         model="mouse_iv", notes="20nm core, near-optimal"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=46.3, zeta_mv=-10.2, peg="no",
         ligand="transferrin", ligand_count=100, kd_nM=0.018,
         brain_metric=0.4, metric_type="relative_brain_counts",
         model="mouse_iv", notes="20nm core, excess avidity reduces brain entry"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=78.1, zeta_mv=-5.3, peg="no",
         ligand="transferrin", ligand_count=20, kd_nM=0.89,
         brain_metric=1.8, metric_type="relative_brain_counts",
         model="mouse_iv", notes="50nm core, best size"),

    dict(paper="Wiley_PNAS_2013", np_type="gold",
         size_nm=85.4, zeta_mv=-6.3, peg="no",
         ligand="transferrin", ligand_count=200, kd_nM=0.014,
         brain_metric=0.7, metric_type="relative_brain_counts",
         model="mouse_iv", notes="50nm core, excess Tf reduces penetration"),

    # ──────────────────────────────────────────────────────────
    # Paper 2: Wiley et al. PNAS 2015
    # "Increased brain uptake of targeted nanoparticles by adding
    #  an acid-cleavable linkage between transferrin and the NP core"
    # DOI: 10.1073/pnas.1517048112
    # NP type: ~75nm gold, normal (N) vs acid-cleavable (C) linker
    # Outcome: % BBB crossing in vitro (2h transwell), in vivo ~1%ID
    # ──────────────────────────────────────────────────────────
    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=75.1, zeta_mv=-5.75, peg="yes",
         ligand="none", ligand_count=0, kd_nM=None,
         brain_metric=10.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="mPEG control, no targeting"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=79.1, zeta_mv=-5.77, peg="yes",
         ligand="transferrin_normal", ligand_count=20, kd_nM=0.408,
         brain_metric=47.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="20Tf normal linkage"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=78.1, zeta_mv=-7.78, peg="yes",
         ligand="transferrin_normal", ligand_count=200, kd_nM=0.029,
         brain_metric=50.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="200Tf normal linkage"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=75.7, zeta_mv=-11.25, peg="yes",
         ligand="transferrin_cleavable", ligand_count=20, kd_nM=0.788,
         brain_metric=61.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="20Tf acid-cleavable linker"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=77.2, zeta_mv=-7.93, peg="yes",
         ligand="transferrin_cleavable", ligand_count=120, kd_nM=0.096,
         brain_metric=93.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="120Tf acid-cleavable linker"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=73.6, zeta_mv=-7.47, peg="yes",
         ligand="transferrin_cleavable", ligand_count=200, kd_nM=0.030,
         brain_metric=94.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="200Tf acid-cleavable linker, best performer, ~1%ID in vivo"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=76.1, zeta_mv=-6.26, peg="yes",
         ligand="anti_TfR_antibody_normal", ligand_count=2, kd_nM=0.441,
         brain_metric=35.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="2 anti-TfR Ab, normal linkage"),

    dict(paper="Wiley_PNAS_2015", np_type="gold",
         size_nm=84.4, zeta_mv=-8.83, peg="yes",
         ligand="anti_TfR_antibody_normal", ligand_count=10, kd_nM=0.039,
         brain_metric=38.0, metric_type="pct_bbb_crossing",
         model="mouse_iv", notes="10 anti-TfR Ab, normal linkage"),

    # ──────────────────────────────────────────────────────────
    # Paper 3: Khung et al. ACS Nano / PMC10534822, 2023
    # "Influence of Surface Ligand Density and Particle Size on
    #  the Penetration of the BBB by Porous Silicon Nanoparticles"
    # Outcome: % BBB transport in transwell (48h)
    # ──────────────────────────────────────────────────────────
    dict(paper="Khung_ACSNano_2023", np_type="porous_silicon",
         size_nm=180.9, zeta_mv=-10.0, peg="yes",
         ligand="none", ligand_count=0, kd_nM=None,
         brain_metric=13.7, metric_type="pct_bbb_crossing",
         model="in_vitro_transwell", notes="mPEG control"),

    dict(paper="Khung_ACSNano_2023", np_type="porous_silicon",
         size_nm=170.1, zeta_mv=-12.0, peg="yes",
         ligand="transferrin", ligand_count=None, kd_nM=None,
         brain_metric=15.0, metric_type="pct_bbb_crossing",
         model="in_vitro_transwell",
         notes="Tf low density (2.10 nmol/mg), Tf:mPEG=1:50"),

    dict(paper="Khung_ACSNano_2023", np_type="porous_silicon",
         size_nm=170.1, zeta_mv=-12.0, peg="yes",
         ligand="transferrin", ligand_count=None, kd_nM=None,
         brain_metric=24.4, metric_type="pct_bbb_crossing",
         model="in_vitro_transwell",
         notes="Tf medium density (3.83 nmol/mg), Tf:mPEG=1:9"),

    dict(paper="Khung_ACSNano_2023", np_type="porous_silicon",
         size_nm=170.1, zeta_mv=-12.0, peg="yes",
         ligand="transferrin", ligand_count=None, kd_nM=None,
         brain_metric=23.6, metric_type="pct_bbb_crossing",
         model="in_vitro_transwell",
         notes="Tf high density (4.92 nmol/mg), full Tf-PEG"),

    dict(paper="Khung_ACSNano_2023", np_type="porous_silicon",
         size_nm=299.3, zeta_mv=-12.0, peg="yes",
         ligand="transferrin", ligand_count=None, kd_nM=None,
         brain_metric=17.5, metric_type="pct_bbb_crossing",
         model="in_vitro_transwell",
         notes="Tf high density, LARGE particles (300nm) → reduced transport"),

    # ──────────────────────────────────────────────────────────
    # Paper 4: Review Table 4 from:
    # Nájera-Maldonado et al. Pharmaceutics 2025
    # "Cracking the Blood-Brain Barrier Code: Rational
    #  Nanomaterial Design for Next-Generation Neurological Therapies"
    # DOI: 10.3390/pharmaceutics17091169 | PMC12473768
    # Table 4: "Comparative data on NPs crossing the BBB"
    # Outcome: %ID/g brain (in vivo IV, unless noted)
    # ──────────────────────────────────────────────────────────
    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="liposome",
         size_nm=90.0, zeta_mv=-10.0, peg="yes",
         ligand="none", ligand_count=0, kd_nM=None,
         brain_metric=0.023, metric_type="pct_ID_brain",
         model="mouse_iv_C57BL6",
         notes="PEGylated liposome, untargeted, 4h post-injection"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="liposome",
         size_nm=90.0, zeta_mv=-10.0, peg="yes",
         ligand="scFv_antibody", ligand_count=None, kd_nM=None,
         brain_metric=0.24, metric_type="pct_ID_brain",
         model="mouse_iv_C57BL6",
         notes="scFv antibody-targeted liposome, 4h, ~10× vs untargeted"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="liposome",
         size_nm=100.0, zeta_mv=5.0, peg="no",
         ligand="TAT_peptide", ligand_count=None, kd_nM=None,
         brain_metric=0.10, metric_type="pct_ID_brain",
         model="mouse_iv",
         notes="TAT peptide-functionalized, cationic, no improved uptake vs control"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="PLGA",
         size_nm=133.0, zeta_mv=-29.0, peg="no",
         ligand="poloxamer188", ligand_count=None, kd_nM=None,
         brain_metric=17.2, metric_type="pct_ID_brain",
         model="rat_C6_glioma_iv",
         notes="PLGA+Poloxamer188 coating, MTX+PTX loaded, 48h, glioma model"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="PLGA",
         size_nm=221.0, zeta_mv=-18.0, peg="no",
         ligand="poloxamer188", ligand_count=None, kd_nM=None,
         brain_metric=17.2, metric_type="pct_ID_brain",
         model="rat_C6_glioma_iv",
         notes="PLGA+Poloxamer188 larger formulation variant"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="PEG_PLGA",
         size_nm=100.0, zeta_mv=-15.0, peg="yes",
         ligand="none", ligand_count=0, kd_nM=None,
         brain_metric=0.5, metric_type="pct_ID_brain",
         model="rodent_iv",
         notes="PEG-PLGA unmodified, minimal brain penetration (<1%ID/g), ~0.5 estimated"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="PAMAM_G4",
         size_nm=4.3, zeta_mv=0.0, peg="no",
         ligand="OH_terminated", ligand_count=None, kd_nM=None,
         brain_metric=1.9, metric_type="ug_per_g_brain",
         model="rat_9L_GL261_tumor",
         notes="PAMAM G4 OH-terminated dendrimer, 24h, 1.9 μg/g tumor"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="PAMAM_G6",
         size_nm=6.7, zeta_mv=0.0, peg="no",
         ligand="OH_terminated", ligand_count=None, kd_nM=None,
         brain_metric=17.6, metric_type="ug_per_g_brain",
         model="mouse_GL261_tumor",
         notes="PAMAM G6 OH-terminated dendrimer, 24h, 17.6 μg/g tumor"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="protein_cage",
         size_nm=12.0, zeta_mv=-8.0, peg="no",
         ligand="H_ferritin", ligand_count=None, kd_nM=None,
         brain_metric=None, metric_type="qualitative_positive",
         model="mouse_iv",
         notes="H-Ferritin nanocage, effective BBB penetration, no exact %ID/g reported"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="VLP",
         size_nm=40.0, zeta_mv=None, peg="no",
         ligand="JC_polyomavirus", ligand_count=None, kd_nM=None,
         brain_metric=0.0, metric_type="pct_ID_brain",
         model="mouse_iv",
         notes="JC polyomavirus VLP, ~0% ID/g brain, not effective"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="gold",
         size_nm=15.0, zeta_mv=0.0, peg="yes",
         ligand="none", ligand_count=0, kd_nM=None,
         brain_metric=0.04, metric_type="pct_ID_brain",
         model="mouse_iv",
         notes="Gold NP PEGylated, no targeting, baseline"),

    dict(paper="Najera_Pharmaceutics_2025_T4", np_type="gold",
         size_nm=15.0, zeta_mv=0.0, peg="yes",
         ligand="anti_JAM_A_antibody", ligand_count=None, kd_nM=None,
         brain_metric=0.13, metric_type="pct_ID_brain",
         model="mouse_iv",
         notes="Gold NP anti-JAM-A antibody, 0.13% ID/g (~3× vs untargeted)"),

]

df = pd.DataFrame(records)

out_dir = "/Users/yangjiawen/Desktop/因果 agent 纳米粒子设计/data"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "np_brain_dataset_starter.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"已生成: {out_path}")
print(f"数据点数: {len(df)}")
print(f"\n各论文数据量:")
print(df["paper"].value_counts())
print(f"\n字段列表: {list(df.columns)}")
print(f"\n前5行:")
print(df.head().to_string())
