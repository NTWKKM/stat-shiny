"""
Sample Size and Power Calculation Library
Implements computations for Study Design:
- Means (T-test)
- Proportions (Chi-sq/Z-test)
- Survival (Log-rank/Cox)
- Correlation
"""
import numpy as np
import statsmodels.stats.power as smp
import statsmodels.stats.proportion as smprop
from scipy import stats

def calculate_power_means(
    n1: float, 
    n2: float | None, 
    mean1: float, 
    mean2: float, 
    sd1: float, 
    sd2: float, 
    alpha: float = 0.05, 
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate Power for Two Independent Means (T-test).
    Returns power (0.0 - 1.0).
    """
    # pooled sd
    # Cohen's d
    if n2 is None: n2 = n1
    
    # Weighted pooled SD
    sd_pooled = np.sqrt( ((n1 - 1)*sd1**2 + (n2 - 1)*sd2**2) / (n1 + n2 - 2) )
    
    effect_size = abs(mean1 - mean2) / sd_pooled
    ratio = n2 / n1
    
    analysis = smp.TTestIndPower()
    power = analysis.solve_power(
        effect_size=effect_size, 
        nobs1=n1, 
        alpha=alpha, 
        ratio=ratio, 
        alternative=alternative
    )
    return float(power)

def calculate_sample_size_means(
    power: float, 
    ratio: float, 
    mean1: float, 
    mean2: float, 
    sd1: float, 
    sd2: float, 
    alpha: float = 0.05, 
    alternative: str = 'two-sided'
) -> dict[str, float]:
    """
    Calculate Sample Size for Two Independent Means.
    Returns dictionary with n1, n2, and total_n.
    """
    # Initial guess for pooled SD (assuming equal N for SD calculation approx, refined by solver)
    # Actually, we need sd_pooled to get effect size. 
    # Approx: sd_pooled ~= sqrt((sd1^2 + sd2^2)/2) for planning
    sd_pooled = np.sqrt((sd1**2 + sd2**2) / 2)
    
    effect_size = abs(mean1 - mean2) / sd_pooled
    
    analysis = smp.TTestIndPower()
    n1 = analysis.solve_power(
        effect_size=effect_size, 
        power=power, 
        alpha=alpha, 
        ratio=ratio, 
        alternative=alternative
    )
    n2 = n1 * ratio
    return {"n1": np.ceil(n1), "n2": np.ceil(n2), "total": np.ceil(n1) + np.ceil(n2)}

def calculate_power_proportions(
    n1: float,
    n2: float | None,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate Power for Two Proportions (Z-test/Chi-sq approx).
    """
    if n2 is None: n2 = n1
    ratio = n2 / n1
    
    # Effect size (h)
    effect_size = smprop.proportion_effectsize(p1, p2)
    
    # Using GofChisquarePower or NormalIndPower
    # NormalIndPower is standard for 2 proportions z-test
    analysis = smp.NormalIndPower()
    power = analysis.solve_power(
        effect_size=effect_size, 
        nobs1=n1, 
        alpha=alpha, 
        ratio=ratio, 
        alternative=alternative
    )
    return float(power)

def calculate_sample_size_proportions(
    power: float,
    ratio: float,
    p1: float,
    p2: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> dict[str, float]:
    """
    Calculate Sample Size for Two Proportions.
    """
    effect_size = smprop.proportion_effectsize(p1, p2)
    
    analysis = smp.NormalIndPower()
    n1 = analysis.solve_power(
        effect_size=effect_size, 
        power=power, 
        alpha=alpha, 
        ratio=ratio, 
        alternative=alternative
    )
    n2 = n1 * ratio
    return {"n1": np.ceil(n1), "n2": np.ceil(n2), "total": np.ceil(n1) + np.ceil(n2)}


def calculate_sample_size_survival(
    power: float,
    ratio: float,
    h0: float, # Hazard Ratio or Median Survival 1
    h1: float, # Median Survival 2 (if using medians) or just HR if param2 is None
    alpha: float = 0.05,
    mode: str = "hr" # "hr" for Hazard Ratio input, "median" for Medians input
) -> dict[str, float]:
    """
    Calculate Sample Size for Log-Rank Test (Survival).
    Freedman's method is commonly used.
    Total Events (E) = ((z_alpha + z_beta)^2 * (1 + ratio*HR)^2) / ( (1-ratio)*HR )? 
    
    Actually simple formula for Total Events (E):
    E = 4 * (z_alpha + z_beta)^2 / ln(HR)^2   (for 1:1 ratio)
    Adjusted for Ratio k=n2/n1:
    E = (z_alpha + z_beta)^2 * ( (1+k)^2 / (k * ln(HR)^2) )
    """
    
    if mode == "median":
        # H0 = m1, H1 = m2. HR = m1/m2 (assuming exp dist)
        hr = h0 / h1
    else:
        hr = h0 # h0 is treated as HR
        
    z_alpha = stats.norm.ppf(1 - alpha/2) # two-sided
    z_beta = stats.norm.ppf(power)
    
    # Schoenberg/Richter formula for Events
    num = (1 + ratio)**2
    den = ratio * (np.log(hr))**2
    total_events = (z_alpha + z_beta)**2 * (num / den)
    
    # We only return required events, N depends on censoring/follow-up which is complex
    # Usually we estimate N assuming probability of event P_event
    # We will return "Total Events Required"
    
    return {"total_events": np.ceil(total_events), "hr": hr}

def calculate_sample_size_correlation(
    power: float,
    r: float,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate Sample Size for Pearson Correlation.
    Use Fisher's Z transformation.
    """
    # Approx N = 3 + ((z_alpha + z_beta) / (0.5 * ln((1+r)/(1-r))))^2
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    c = 0.5 * np.log((1 + r) / (1 - r))
    n = 3 + ((z_alpha + z_beta) / c)**2
    return np.ceil(n)
