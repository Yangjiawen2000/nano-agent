"""
PBPK Simulator for Brain-Targeted Nanoparticle Design
=====================================================
模型描述受体介导转胞吞（Receptor-Mediated Transcytosis, RMT）机制：
  血液循环 → 受体结合 → 内化 → 胞内囊泡 → 转胞吞进入脑实质

状态变量（均为无量纲，归一化为初始血液浓度 = 1.0）：
  NP_blood   : 血液循环中的NP
  R_free     : 游离BBB受体占比 (0~1)
  NP_R       : 受体结合态NP
  NP_vesicle : 内吞囊泡中的NP
  NP_brain   : 到达脑实质的NP（累积量）

参考文献：
  Wiley PNAS 2013  (DOI: 10.1073/pnas.1307152110)
  Wiley PNAS 2015  (DOI: 10.1073/pnas.1517048112)
  Khung ACS Nano 2023 (DOI: 10.1021/acsnano.3c10674)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 1.  参数数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class PBPKParams:
    """PBPK模型动力学参数（时间单位：小时）"""
    k_bind:    float  # 有效结合速率常数 [h⁻¹]，= k_on × R_total（已归一化）
    k_off:     float  # 受体解离速率 [h⁻¹]
    k_endo:    float  # 内吞速率 [h⁻¹]
    k_trans:   float  # 转胞吞速率 [h⁻¹]
    k_lyso:    float  # 溶酶体降解速率 [h⁻¹]
    k_recycle: float  # 受体再循环速率 [h⁻¹]
    CL:        float  # 血液清除速率 [h⁻¹]（包含MPS吸收）

    def summary(self):
        t_half_blood = np.log(2) / self.CL if self.CL > 0 else np.inf
        trans_efficiency = self.k_trans / (self.k_trans + self.k_lyso)
        print(f"  k_bind={self.k_bind:.3f}  k_off={self.k_off:.3f}  "
              f"k_endo={self.k_endo:.3f}  k_trans={self.k_trans:.3f}")
        print(f"  k_lyso={self.k_lyso:.3f}  k_recycle={self.k_recycle:.3f}  "
              f"CL={self.CL:.3f}")
        print(f"  → 血液半衰期: {t_half_blood:.2f} h")
        print(f"  → 转胞吞效率（vs溶酶体）: {trans_efficiency*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 2.  参数映射函数（NP设计参数 → PBPK动力学参数）
# ═══════════════════════════════════════════════════════════════

def _size_binding_factor(size_nm: float) -> float:
    """
    粒径对受体结合速率的影响。
    来源：Wiley PNAS 2013 (~80nm最优）；Khung ACS Nano 2023（>200nm显著降低）
    形状：高斯峰，峰值在80nm，σ=65nm（右尾宽，允许100-150nm也有效）
    """
    factor = np.exp(-((size_nm - 80.0) / 65.0) ** 2)
    return float(np.clip(factor, 0.05, 1.0))


def _ligand_density_factor(ligand_density) -> float:
    """
    配体密度对结合速率的影响（亲和力优化 vs 亲合力陷阱）。
    来源：Wiley PNAS 2013（过高Tf密度→低Kd→脑内递送反而降低）
    - 中等密度最优
    - very_high 密度因"亲合力陷阱"效应下降（NP紧锁在受体上，无法释放进行转胞吞）
    """
    if isinstance(ligand_density, (int, float)):
        # 数值型（每个NP的配体数量）
        n = float(ligand_density)
        if n <= 0:   return 0.02
        if n <= 5:   return 0.40
        if n <= 20:  return 0.85
        if n <= 50:  return 1.00   # 最优区间
        if n <= 100: return 1.20
        return 0.80  # >100，亲合力陷阱
    density_map = {
        'none':      0.02,
        'low':       0.45,
        'medium':    1.00,
        'high':      1.25,
        'very_high': 0.75,   # 亲合力陷阱
    }
    return density_map.get(str(ligand_density).strip().lower(), 1.00)


def map_k_bind(size_nm: float,
               ligand_density,
               ligand_type: str) -> float:
    """
    有效受体结合速率常数 k_bind [h⁻¹]
    = k_on × R_total（已将受体浓度归一化进参数）

    基准值（80nm, transferrin, medium density）= 0.80 h⁻¹
    来源：Wiley PNAS 2013/2015 实验数据拟合
    """
    ligand_base = {
        'transferrin':           0.80,
        'transferrin_normal':    0.80,
        'transferrin_cleavable': 0.85,  # 酸裂解连接体结合略强
        'lactoferrin':           0.70,  # LfR脑内表达量略低于TfR
        'anti_tfr':              1.00,  # 抗体亲和力更高
        'anti_tfr_antibody':     1.00,
        'anti_jam_a_antibody':   0.60,  # 不同受体
        'scfv_antibody':         0.90,
        'h_ferritin':            0.50,
        'none':                  0.02,  # 非特异性
    }
    key = str(ligand_type).strip().lower().replace(' ', '_')
    base = ligand_base.get(key, 0.40)
    return base * _size_binding_factor(size_nm) * _ligand_density_factor(ligand_density)


def map_k_off(ligand_type: str) -> float:
    """
    受体解离速率 k_off [h⁻¹]
    决定NP在受体上的停留时间。
    来源：Kd = k_off / k_bind；Wiley PNAS 2013 Kd测量值反推
    """
    koff_map = {
        'transferrin':           0.40,
        'transferrin_normal':    0.40,
        'transferrin_cleavable': 0.35,
        'lactoferrin':           0.35,
        'anti_tfr':              0.20,  # 抗体解离更慢
        'anti_tfr_antibody':     0.20,
        'anti_jam_a_antibody':   0.30,
        'scfv_antibody':         0.25,
        'h_ferritin':            0.50,
        'none':                  0.60,
    }
    key = str(ligand_type).strip().lower().replace(' ', '_')
    return koff_map.get(key, 0.40)


def map_k_trans(zeta_mv: float) -> float:
    """
    转胞吞速率 k_trans [h⁻¹]
    ζ电位影响：-10~-20 mV为最优区间（增强内体膜破坏，促进转胞吞）
    来源：Khung ACS Nano 2023；Bhattacharya 2023（综述数据）

    高斯峰：峰值在 -15 mV，σ = 10 mV
    范围：0.01（无效）~ 0.10 h⁻¹（最优）
    """
    k_max = 0.10   # 最优ζ电位下的转胞吞速率上限
    optimal_zeta = -15.0
    sigma = 10.0
    factor = np.exp(-((zeta_mv - optimal_zeta) / sigma) ** 2)
    return float(np.clip(k_max * factor, 0.008, k_max))


def map_k_lyso(zeta_mv: float) -> float:
    """
    溶酶体降解速率 k_lyso [h⁻¹]
    大多数内吞NP走溶酶体降解路径（k_lyso >> k_trans）。
    极负ζ电位增加蛋白冠，可能稍微增加溶酶体摄取。
    """
    k_base = 0.20
    # 轻微ζ电位效应：偏离最优区间时溶酶体略增
    penalty = 0.005 * abs(zeta_mv + 15.0)
    return float(np.clip(k_base + penalty, 0.15, 0.35))


def map_CL(size_nm: float, peg: str) -> float:
    """
    血液清除速率 CL [h⁻¹]（网状内皮系统/MPS摄取 + 肾滤过）
    - 大粒径：更快被MPS清除
    - PEG修饰：显著延长循环时间（减少MPS识别）

    参考值：PEG化80nm NP的t1/2 ≈ 1.5~3h（CL ≈ 0.23~0.46 h⁻¹）
    来源：综述数据（Nájera Pharmaceutics 2025）
    """
    # 尺寸效应（幂律）
    size_factor = (size_nm / 80.0) ** 0.4
    # PEG效应
    peg_str = str(peg).strip().lower()
    if peg_str in ('yes', 'true', '1', 'peg'):
        peg_factor = 0.45  # PEG使清除减少55%
    else:
        peg_factor = 1.00
    CL_ref = 0.50  # 基准：80nm无PEG时的清除速率
    return float(np.clip(CL_ref * size_factor * peg_factor, 0.05, 2.0))


def map_drug_loading_factor(drug_loading: float) -> float:
    """
    药物有效载量倍增因子（% w/w）。
    10% 载量 → factor=1.0；>20% 时配方稳定性下降 → factor 衰减。
    来源：Stylianopoulos Expert Opin Drug Deliv 2013（载量-稳定性权衡）
    """
    dl = max(0.0, float(drug_loading))
    payload = min(dl / 10.0, 1.5)
    stability = 1.0 if dl <= 20 else max(0.5, 1.0 - (dl - 20) * 0.025)
    return float(payload * stability)


def map_shape_factors(particle_shape: str) -> tuple:
    """
    粒子形状对 (CL_factor, k_bind_factor) 的影响。
    杆形/蠕虫形：血管边缘化增强、循环时间延长（↓CL ~30%），
    但与受体的接触面积/方向受限（↓k_bind 约12%）。
    来源：Decuzzi PNAS 2010；Geng Nature Nano 2007（filomicelles）
    """
    shape = str(particle_shape).strip().lower()
    return {
        'sphere': (1.00, 1.00),
        'rod':    (0.70, 0.88),
        'disk':   (0.82, 0.93),
        'worm':   (0.65, 0.80),
    }.get(shape, (1.00, 1.00))


def map_surface_chemistry_factors(hydrophobicity: float,
                                   surface_coating: str) -> tuple:
    """
    表面化学对 (CL_factor, k_endo_factor) 的影响。
    疏水性 → 蛋白冠形成 → 吞噬细胞识别 → ↑CL（二次方关系）。
    来源：Walkey ACS Nano 2014（蛋白吸附∝疏水性²）
    两性离子涂层：近零蛋白吸附 → ↓CL ~35%。
    脂质涂层：膜融合 → 内体逃逸增强 → ↑k_endo。
    来源：Schlenoff Langmuir 2014；Semple Nature Biotech 2010
    """
    h = float(np.clip(hydrophobicity, 0.0, 1.0))
    CL_hydro = 1.0 + 0.6 * h ** 2

    cl_coat, endo_coat = {
        'peg':          (0.90, 1.00),
        'lipid':        (0.95, 1.25),
        'zwitterionic': (0.65, 0.90),
        'polymer':      (0.88, 1.00),
        'none':         (1.00, 1.00),
    }.get(str(surface_coating).strip().lower(), (1.00, 1.00))
    return float(CL_hydro * cl_coat), float(endo_coat)


def design_to_params(size_nm:          float,
                     zeta_mv:          float,
                     peg,
                     ligand_type:      str,
                     ligand_density,
                     particle_shape:   str   = 'sphere',
                     hydrophobicity:   float = 0.0,
                     surface_coating:  str   = 'none',
                     k_endo:    float = 0.12,
                     k_recycle: float = 0.08) -> PBPKParams:
    """
    NP设计参数 → PBPKParams

    固定参数（文献来源）：
      k_endo = 0.12 h⁻¹  受体内吞速率（多数文献报告 0.08~0.15 h⁻¹）
      k_recycle = 0.08 h⁻¹  受体再循环（维持受体池）
    """
    shape_cl,  shape_kb  = map_shape_factors(particle_shape)
    chem_cl,   chem_endo = map_surface_chemistry_factors(hydrophobicity, surface_coating)
    return PBPKParams(
        k_bind    = map_k_bind(size_nm, ligand_density, ligand_type) * shape_kb,
        k_off     = map_k_off(ligand_type),
        k_endo    = k_endo * chem_endo,
        k_trans   = map_k_trans(zeta_mv),
        k_lyso    = map_k_lyso(zeta_mv),
        k_recycle = k_recycle,
        CL        = map_CL(size_nm, peg) * shape_cl * chem_cl,
    )


# ═══════════════════════════════════════════════════════════════
# 3.  ODE 系统
# ═══════════════════════════════════════════════════════════════

def pbpk_ode(t, y, p: PBPKParams):
    """
    受体介导转胞吞PBPK ODE系统（5个状态变量）

    y[0] NP_blood   : 血液中NP（归一化，初值=1.0）
    y[1] R_free     : 游离受体比例（初值=1.0）
    y[2] NP_R       : 受体结合态NP
    y[3] NP_vesicle : 内体囊泡中NP
    y[4] NP_brain   : 脑实质累积NP

    速率方程（各速率单位 [h⁻¹]）：
      binding      = k_bind  × NP_blood × R_free
      unbinding    = k_off   × NP_R
      endocytosis  = k_endo  × NP_R
      transcytosis = k_trans × NP_vesicle
      lysosomal    = k_lyso  × NP_vesicle
      recycling    = k_recycle × (1 - R_free)   ← 占用受体再循环
      clearance    = CL × NP_blood
    """
    NP_blood, R_free, NP_R, NP_vesicle, NP_brain = y

    # 数值守卫（防止负值漂移）
    NP_blood   = max(NP_blood,   0.0)
    R_free     = np.clip(R_free, 0.0, 1.0)
    NP_R       = max(NP_R,       0.0)
    NP_vesicle = max(NP_vesicle, 0.0)

    binding      = p.k_bind    * NP_blood * R_free
    unbinding    = p.k_off     * NP_R
    endocytosis  = p.k_endo    * NP_R
    transcytosis = p.k_trans   * NP_vesicle
    lysosomal    = p.k_lyso    * NP_vesicle
    recycling    = p.k_recycle * (1.0 - R_free)
    clearance    = p.CL        * NP_blood

    dNP_blood   = -binding + unbinding - clearance
    dR_free     = -binding + unbinding + recycling
    dNP_R       =  binding - unbinding - endocytosis
    dNP_vesicle =  endocytosis - transcytosis - lysosomal
    dNP_brain   =  transcytosis   # 脑内NP累积（不可逆）

    return [dNP_blood, dR_free, dNP_R, dNP_vesicle, dNP_brain]


# ═══════════════════════════════════════════════════════════════
# 4.  主仿真函数
# ═══════════════════════════════════════════════════════════════

def pbpk_simulate(size_nm:          float,
                  zeta_mv:          float,
                  peg,
                  ligand_type:      str,
                  ligand_density,
                  drug_loading:     float = 0.0,
                  particle_shape:   str   = 'sphere',
                  hydrophobicity:   float = 0.0,
                  surface_coating:  str   = 'none',
                  patient_type:     str   = 'adult_healthy',
                  t_span:           tuple = (0, 24),
                  n_points:         int   = 300,
                  verbose:          bool  = False) -> dict:
    """
    输入：NP设计参数（+可选：药物载量、粒子形状、表面化学、虚拟患者）
    输出：脑内时程曲线 + AUC

    参数
    ----
    size_nm         : 流体力学直径 (nm)
    zeta_mv         : ζ电位 (mV)
    peg             : PEG修饰 ('yes'/'no')
    ligand_type     : 配体种类
    ligand_density  : 配体密度（每NP的配体数）
    drug_loading    : 药物载量 (% w/w，0=未指定)，影响 AUC_drug_brain 输出
    particle_shape  : 粒子形状 ('sphere'/'rod'/'disk'/'worm')
    hydrophobicity  : 表面疏水性 (0–1)，影响蛋白冠/MPS清除
    surface_coating : 表面涂层 ('none'/'peg'/'lipid'/'zwitterionic'/'polymer')
    patient_type    : 虚拟患者 ('adult_healthy'/'pediatric_healthy'/... 见 virtual_patients.py)
    t_span          : 仿真时间区间 (h)
    n_points        : 时间采样点数
    verbose         : 打印参数摘要
    """
    params = design_to_params(size_nm, zeta_mv, peg, ligand_type, ligand_density,
                               particle_shape, hydrophobicity, surface_coating)

    if patient_type not in ('adult_healthy', 'healthy_adult', 'adult', ''):
        try:
            from virtual_patients import apply_patient_scaling
            params = apply_patient_scaling(params, patient_type)
        except Exception:
            pass

    if verbose:
        print(f"\n── PBPK参数（{size_nm}nm, ζ={zeta_mv}mV, PEG={peg}, "
              f"{ligand_type}, density={ligand_density}, "
              f"shape={particle_shape}, patient={patient_type}）──")
        params.summary()

    y0 = [1.0, 1.0, 0.0, 0.0, 0.0]  # NP_blood=1, R_free=1, 其余=0
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        pbpk_ode,
        t_span, y0,
        args=(params,),
        method='Radau',   # 刚性ODE求解器
        t_eval=t_eval,
        rtol=1e-7, atol=1e-10,
        dense_output=False,
    )

    if not sol.success:
        print(f"[警告] ODE求解失败：{sol.message}")

    C_brain  = sol.y[4]
    C_blood  = sol.y[0]
    AUC      = float(np.trapezoid(C_brain, sol.t))
    AUC_blood = float(np.trapezoid(C_blood, sol.t))
    AUC_ratio = AUC / AUC_blood if AUC_blood > 1e-10 else 0.0

    drug_factor   = map_drug_loading_factor(drug_loading) if drug_loading > 0 else 0.0
    AUC_drug_brain = round(AUC * drug_factor, 6)

    return {
        't':            sol.t,
        'C_brain':      C_brain,
        'C_blood':      C_blood,
        'R_free':       sol.y[1],
        'NP_R':         sol.y[2],
        'NP_vesicle':   sol.y[3],
        'AUC':          AUC,
        'AUC_ratio':    AUC_ratio,
        'AUC_drug_brain': AUC_drug_brain,
        'params':       params,
        'success':      sol.success,
        'patient_type': patient_type,
    }


# ═══════════════════════════════════════════════════════════════
# 5.  定性验证（基于已知文献趋势）
# ═══════════════════════════════════════════════════════════════

def validate_trends(plot: bool = True):
    """
    验证仿真器是否重现文献中的已知定性趋势：
    1. 80nm > 20nm ≈ 200nm（粒径效应，Wiley PNAS 2013）
    2. Tf靶向 > 无靶向（配体效应）
    3. ζ = -15mV > ζ = 0mV > ζ = -30mV（ζ电位效应）
    4. 中等配体密度 > 极高密度（亲合力陷阱，Wiley PNAS 2013）
    5. PEG修饰 > 无PEG（循环时间效应）
    """
    base = dict(zeta_mv=-12, peg='yes', ligand_type='transferrin',
                ligand_density='medium')

    tests = [
        # (描述, 参数覆盖)
        ("20nm",  dict(**base, size_nm=20)),
        ("80nm",  dict(**base, size_nm=80)),
        ("150nm", dict(**base, size_nm=150)),
        ("200nm", dict(**base, size_nm=200)),
    ]
    print("═" * 60)
    print("验证1：粒径效应（期望: 80nm最优）")
    _run_tests(tests)

    tests = [
        ("无靶向",    dict(size_nm=80, zeta_mv=-12, peg='yes',
                        ligand_type='none', ligand_density='none')),
        ("Tf-中等",   dict(size_nm=80, zeta_mv=-12, peg='yes',
                        ligand_type='transferrin', ligand_density='medium')),
        ("Tf-高",     dict(size_nm=80, zeta_mv=-12, peg='yes',
                        ligand_type='transferrin', ligand_density='high')),
        ("Tf-极高",   dict(size_nm=80, zeta_mv=-12, peg='yes',
                        ligand_type='transferrin', ligand_density='very_high')),
        ("Lf-中等",   dict(size_nm=80, zeta_mv=-12, peg='yes',
                        ligand_type='lactoferrin', ligand_density='medium')),
    ]
    print("\n验证2：配体种类与密度（期望: 中等密度>极高密度，Tf>无靶向）")
    _run_tests(tests)

    tests = [
        ("ζ=+5mV",  dict(size_nm=80, peg='yes', ligand_type='transferrin',
                        ligand_density='medium', zeta_mv=5)),
        ("ζ=-5mV",  dict(size_nm=80, peg='yes', ligand_type='transferrin',
                        ligand_density='medium', zeta_mv=-5)),
        ("ζ=-15mV", dict(size_nm=80, peg='yes', ligand_type='transferrin',
                        ligand_density='medium', zeta_mv=-15)),
        ("ζ=-25mV", dict(size_nm=80, peg='yes', ligand_type='transferrin',
                        ligand_density='medium', zeta_mv=-25)),
        ("ζ=-35mV", dict(size_nm=80, peg='yes', ligand_type='transferrin',
                        ligand_density='medium', zeta_mv=-35)),
    ]
    print("\n验证3：ζ电位效应（期望: -15mV最优）")
    _run_tests(tests)

    tests = [
        ("无PEG", dict(size_nm=80, zeta_mv=-15, peg='no',
                     ligand_type='transferrin', ligand_density='medium')),
        ("有PEG", dict(size_nm=80, zeta_mv=-15, peg='yes',
                     ligand_type='transferrin', ligand_density='medium')),
    ]
    print("\n验证4：PEG修饰效应（期望: PEG > 无PEG）")
    _run_tests(tests)

    if plot:
        _plot_sensitivity()


def _run_tests(tests):
    print(f"  {'设计':<18} {'AUC_brain':>10} {'AUC_ratio':>10} {'排名'}")
    results = []
    for name, kwargs in tests:
        r = pbpk_simulate(**kwargs)
        results.append((name, r['AUC'], r['AUC_ratio']))
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    rank_map = {name: i+1 for i, (name, _, _) in enumerate(results_sorted)}
    for name, auc, ratio in results:
        print(f"  {name:<18} {auc:>10.5f} {ratio:>10.4f}   #{rank_map[name]}")


def _plot_sensitivity():
    """绘制三个核心设计参数的敏感性曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PBPK仿真器参数敏感性分析", fontsize=13, fontweight='bold')

    # 图1：粒径
    sizes = np.linspace(10, 300, 50)
    aucs = [pbpk_simulate(s, -15, 'yes', 'transferrin', 'medium')['AUC']
            for s in sizes]
    axes[0].plot(sizes, aucs, 'b-', lw=2)
    axes[0].axvline(80, color='r', ls='--', alpha=0.6, label='80nm（最优）')
    axes[0].set_xlabel('粒径 (nm)')
    axes[0].set_ylabel('AUC_brain (归一化)')
    axes[0].set_title('粒径效应')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 图2：ζ电位
    zetas = np.linspace(-40, 10, 60)
    aucs_z = [pbpk_simulate(80, z, 'yes', 'transferrin', 'medium')['AUC']
              for z in zetas]
    axes[1].plot(zetas, aucs_z, 'g-', lw=2)
    axes[1].axvline(-15, color='r', ls='--', alpha=0.6, label='-15mV（最优）')
    axes[1].set_xlabel('ζ电位 (mV)')
    axes[1].set_title('ζ电位效应')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 图3：典型设计的时程曲线对比
    designs = [
        ("最优设计(80nm,-15mV,Tf,中)",
         pbpk_simulate(80,  -15, 'yes', 'transferrin',  'medium')),
        ("大粒径(200nm,-15mV,Tf,中)",
         pbpk_simulate(200, -15, 'yes', 'transferrin',  'medium')),
        ("无靶向(80nm,-15mV,无)",
         pbpk_simulate(80,  -15, 'yes', 'none', 'none')),
        ("无PEG(80nm,-15mV,Tf,中)",
         pbpk_simulate(80,  -15, 'no',  'transferrin',  'medium')),
    ]
    for label, r in designs:
        axes[2].plot(r['t'], r['C_brain'] * 100, lw=2, label=label)
    axes[2].set_xlabel('时间 (h)')
    axes[2].set_ylabel('C_brain (% 初始剂量)')
    axes[2].set_title('典型设计时程曲线对比')
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/yangjiawen/Desktop/因果 agent 纳米粒子设计/data/pbpk_sensitivity.png',
                dpi=150, bbox_inches='tight')
    print("\n  [已保存] data/pbpk_sensitivity.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 6.  与数据集的定量对比
# ═══════════════════════════════════════════════════════════════

def compare_with_dataset(csv_path: str, metric_filter: str = 'pct_bbb_crossing'):
    """
    将仿真结果与数据集中同类指标的实验值进行相关性对比。
    仅比较相同metric_type的行（不同指标量纲不同，不能混用）。
    """
    df = pd.read_csv(csv_path)
    df_sub = df[df['metric_type'] == metric_filter].dropna(
        subset=['size_nm', 'zeta_mv', 'ligand', 'brain_metric'])

    if df_sub.empty:
        print(f"未找到 metric_type='{metric_filter}' 的行")
        return

    print(f"\n与数据集对比（metric_type={metric_filter}，共{len(df_sub)}行）")
    print(f"{'来源':<30} {'粒径':>6} {'ζ(mV)':>7} {'配体':<22} "
          f"{'实验值':>8} {'仿真AUC':>9} {'排名一致'}")
    print("-" * 95)

    rows = []
    for _, row in df_sub.iterrows():
        lig = str(row.get('ligand', 'none')).strip()
        dens = row.get('ligand_count') if pd.notna(row.get('ligand_count')) else 'medium'
        peg  = str(row.get('peg', 'no'))
        r = pbpk_simulate(
            size_nm        = float(row['size_nm']),
            zeta_mv        = float(row['zeta_mv']),
            peg            = peg,
            ligand_type    = lig,
            ligand_density = dens,
        )
        rows.append((row, r['AUC']))

    # 排名相关
    exp_vals  = [float(r[0]['brain_metric']) for r in rows]
    sim_vals  = [r[1] for r in rows]
    exp_rank  = pd.Series(exp_vals).rank(ascending=False).tolist()
    sim_rank  = pd.Series(sim_vals).rank(ascending=False).tolist()

    n_agree = sum(1 for e, s in zip(exp_rank, sim_rank) if abs(e - s) <= 1)

    for i, (row_data, auc_sim) in enumerate(rows):
        agree = "✓" if abs(exp_rank[i] - sim_rank[i]) <= 1 else "✗"
        print(f"  {str(row_data['paper']):<28} {row_data['size_nm']:>6.0f} "
              f"{row_data['zeta_mv']:>7.1f} {str(row_data['ligand']):<22} "
              f"{row_data['brain_metric']:>8.2f} {auc_sim:>9.5f}  {agree}")

    # 斯皮尔曼秩相关
    from scipy.stats import spearmanr
    rho, pval = spearmanr(exp_vals, sim_vals)
    print(f"\n  Spearman ρ = {rho:.3f}  (p = {pval:.3f})")
    print(f"  排名偏差 ≤1 的比例：{n_agree}/{len(rows)} = {n_agree/len(rows)*100:.0f}%")
    return rho


# ═══════════════════════════════════════════════════════════════
# 7.  快速预测接口（供 Agent 调用）
# ═══════════════════════════════════════════════════════════════

def predict_brain_delivery(design: dict) -> dict:
    """
    Agent工具接口：输入设计字典，返回预测结果。

    design 字典示例：
        {'size_nm': 80, 'zeta_mv': -15, 'peg': 'yes',
         'ligand_type': 'transferrin', 'ligand_density': 'medium'}
    """
    required = ['size_nm', 'zeta_mv', 'peg', 'ligand_type', 'ligand_density']
    for key in required:
        if key not in design:
            raise ValueError(f"design 字典缺少必要字段：'{key}'")

    result = pbpk_simulate(**design)

    return {
        'AUC_brain':      round(result['AUC'],            6),
        'AUC_ratio':      round(result['AUC_ratio'],       4),
        'AUC_drug_brain': round(result['AUC_drug_brain'],  6),
        'C_brain_peak':   round(float(result['C_brain'].max()), 6),
        't_peak_h':       round(float(result['t'][result['C_brain'].argmax()]), 2),
        'patient_type':   result.get('patient_type', 'adult_healthy'),
        'params': {
            'k_bind':   round(result['params'].k_bind,   3),
            'k_off':    round(result['params'].k_off,    3),
            'k_trans':  round(result['params'].k_trans,  4),
            'k_lyso':   round(result['params'].k_lyso,   4),
            'CL':       round(result['params'].CL,       3),
        }
    }


# ═══════════════════════════════════════════════════════════════
# 8.  入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("PBPK 脑靶向纳米粒子仿真器")
    print("=" * 60)

    # ── 单次仿真示例 ──
    print("\n【示例1】参考设计（80nm, Tf, -15mV, PEG）")
    result = pbpk_simulate(
        size_nm=80, zeta_mv=-15, peg='yes',
        ligand_type='transferrin', ligand_density='medium',
        verbose=True
    )
    print(f"  AUC_brain  = {result['AUC']:.5f}（归一化）")
    print(f"  AUC_ratio  = {result['AUC_ratio']:.4f}（脑/血浆）")
    print(f"  C_brain峰值= {result['C_brain'].max()*100:.3f}% 初始剂量")
    print(f"  峰值时间   = {result['t'][result['C_brain'].argmax()]:.2f} h")

    # ── Agent接口示例 ──
    print("\n【示例2】Agent工具接口调用")
    pred = predict_brain_delivery({
        'size_nm': 80, 'zeta_mv': -15, 'peg': 'yes',
        'ligand_type': 'transferrin', 'ligand_density': 'medium'
    })
    print(f"  {pred}")

    # ── 定性验证 ──
    print("\n【定性趋势验证】")
    validate_trends(plot=True)

    # ── 与数据集对比 ──
    csv_path = '/Users/yangjiawen/Desktop/因果 agent 纳米粒子设计/data/np_brain_dataset_final.csv'
    compare_with_dataset(csv_path, metric_filter='pct_bbb_crossing')
