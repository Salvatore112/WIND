import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.colors as mcolors
import warnings
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLE
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# -----------------------------------------------------------------------------
# 2. MAIN VISUALIZER CLASS
# -----------------------------------------------------------------------------
class ComprehensiveVisualizer:
    def __init__(self, filename="full_results.csv"):
        self.output_dir = Path("wind_comprehensive_analysis")
        self.output_dir.mkdir(exist_ok=True)

        # --- Auto-detect file location ---
        candidates = [
            Path(filename),
            Path("results_lyapunov") / filename,
            Path("results") / filename,
            Path(__file__).parent / filename,
            Path(__file__).parent / "results_lyapunov" / filename,
        ]

        self.file_path = next((p for p in candidates if p.exists()), None)

        if not self.file_path:
            print(
                f"❌ Error: Could not find '{filename}' in current or standard subfolders."
            )
            exit(1)

        print(f"📂 Loading data from: {self.file_path.absolute()}...")
        self.df = self._load_data()
        print(f"✅ Loaded {len(self.df)} runs after filtering.")

    def _load_data(self):
        try:
            df = pd.read_csv(self.file_path)

            # --- FILTERING: REMOVE REDUNDANT ALGORITHMS ---
            # Removing KieferWolfowitz and FiniteDiffCentral as requested
            excluded_algos = ["KieferWolfowitz", "FiniteDiffCentral"]
            df = df[~df["algorithm"].isin(excluded_algos)]

            # Numeric conversion
            cols = ["lyapunov", "A", "rho", "runtime", "is_stabilized"]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # --- STRICT CLASSIFICATION ---
            first_order_set = {
                "SGD",
                "SGD_Polyak",
                "HeavyBall",
                "Nesterov",
                "Adam",
                "AdamW",
                "AMSGrad",
                "SMD",
                "RDA",
                "ProxSGD",
                "AdaptiveLR",
                "SignSGD",
            }

            # Note: Removed excluded algos from this set
            zero_order_set = {
                "RandomSearch",
                "OnePointSPSA",
                "FDSA",
                "SPSA",
                "ZOSGD",
                "ZOSignSGD",
                "QuadraticInterpolation",
                "NedicSubgradient",
                "AcceleratedSPSA",
                "CMAES",
                "GPUCB",
            }

            def classify_algo(name):
                clean_name = str(name).strip()
                if clean_name in first_order_set:
                    return "First-Order"
                elif clean_name in zero_order_set:
                    return "Zero-Order"
                else:
                    # Fallback
                    if any(
                        x in clean_name
                        for x in ["SPSA", "ZO", "Search", "CMA", "Finite", "Kiefer"]
                    ):
                        return "Zero-Order"
                    return "First-Order"

            df["Type"] = df["algorithm"].apply(classify_algo)

            # Fill stabilization bool
            df["is_stabilized"] = df["is_stabilized"].fillna(0).astype(bool)

            return df
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            exit(1)

    # -------------------------------------------------------------------------
    # PLOT 1: STABILIZATION RATES (Definition 1)
    # -------------------------------------------------------------------------
    def plot_stabilization_rates(self):
        print("📊 1. Generating Stabilization Rates...")

        # Calculate success rate per Algo/Rho
        agg = (
            self.df.groupby(["algorithm", "rho", "Type"])["is_stabilized"]
            .mean()
            .reset_index()
        )

        # Split plots by Type
        for algo_type in ["First-Order", "Zero-Order"]:
            subset = agg[agg["Type"] == algo_type]
            if subset.empty:
                continue

            g = sns.catplot(
                data=subset,
                x="algorithm",
                y="is_stabilized",
                col="rho",
                kind="bar",
                palette="viridis",
                height=5,
                aspect=1.2,
                sharex=False,
            )

            g.set_axis_labels("", "Stabilization Rate (0.0 - 1.0)")
            g.set_titles(f"{algo_type} | Regime $\\rho = {{col_name}}$")

            for ax in g.axes.flat:
                ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Target")
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=90, ha="right", fontsize=10
                )
                ax.grid(axis="y", linestyle="--", alpha=0.3)

            g.fig.suptitle(
                f"Fig 1 ({algo_type}): Verification of Definition 1",
                y=1.05,
                weight="bold",
            )
            g.savefig(self.output_dir / f"1_Stabilization_Rates_{algo_type}.png")
            plt.close()

    # -------------------------------------------------------------------------
    # PLOT 2: CONVERGENCE BOUND HEATMAPS (95th Percentile)
    # -------------------------------------------------------------------------
    def plot_bound_heatmaps(self):
        print("📊 2. Generating Convergence Bound Heatmaps...")

        for algo_type in ["First-Order", "Zero-Order"]:
            subset = self.df[self.df["Type"] == algo_type].copy()
            if subset.empty:
                continue

            # 95th Percentile of Lyapunov Error
            grouped = (
                subset.groupby(["algorithm", "rho", "A"])["lyapunov"]
                .quantile(0.95)
                .reset_index()
            )
            grouped["Regime"] = grouped.apply(
                lambda x: f"ρ={x.rho}\nA={x.A:.3f}", axis=1
            )
            grouped.sort_values(by=["rho", "A"], inplace=True)

            pivot = grouped.pivot(
                index="algorithm", columns="Regime", values="lyapunov"
            )

            # Sort by quality
            pivot["score"] = pivot.median(axis=1)
            pivot = pivot.sort_values("score")
            pivot.drop(columns=["score"], inplace=True)

            plt.figure(figsize=(18, max(6, len(pivot) * 0.8)))

            # Annotations format
            annot = pivot.applymap(lambda x: f"{x:.1f}" if x < 1000 else f"{x:.0e}")

            sns.heatmap(
                pivot,
                annot=annot,
                fmt="",
                cmap="RdYlGn_r",
                norm=mcolors.LogNorm(vmin=max(1e-4, pivot.min().min()), vmax=1000),
                cbar_kws={"label": "95% Convergence Bound (Log Scale)"},
                linewidths=0.5,
                linecolor="white",
            )

            plt.title(
                f"Fig 2: Convergence Bounds (95% CI) - {algo_type}\n(Green = Stable/Low Error)",
                fontsize=16,
                weight="bold",
                pad=20,
            )
            plt.xlabel("Regime (Smoothness ρ, Drift A)")
            plt.ylabel(None)
            plt.xticks(rotation=90)

            plt.savefig(self.output_dir / f"2_Bound_Heatmap_{algo_type}.png")
            plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3: CONVERGENCE PROFILES (Drift Sensitivity)
    # -------------------------------------------------------------------------
    def plot_convergence_profiles(self):
        print("📊 3. Generating Convergence Profiles (Drift Sensitivity)...")

        for algo_type in ["First-Order", "Zero-Order"]:
            subset = self.df[
                (self.df["Type"] == algo_type) & (self.df["lyapunov"] < 1e10)
            ]
            if subset.empty:
                continue

            g = sns.FacetGrid(subset, col="rho", height=5, aspect=1.1, sharey=False)
            g.map_dataframe(
                sns.lineplot,
                x="A",
                y="lyapunov",
                hue="algorithm",
                style="algorithm",
                markers=True,
                dashes=False,
                linewidth=2,
                estimator=np.median,
            )

            g.set(xscale="log", yscale="log")
            g.set_axis_labels("Drift Magnitude A (Log)", "Median Lyapunov Error (Log)")
            g.add_legend(
                title="Algorithm", bbox_to_anchor=(1.02, 0.5), loc="center left"
            )

            plt.subplots_adjust(top=0.85)
            g.fig.suptitle(
                f"Fig 3: Drift Sensitivity Profile ({algo_type})\n(Flat line = Ideal Invariance)",
                fontsize=16,
                weight="bold",
            )

            g.savefig(self.output_dir / f"3_Profile_{algo_type}.png")
            plt.close()

    # -------------------------------------------------------------------------
    # PLOT 4: DOLAN-MORE PERFORMANCE PROFILES
    # -------------------------------------------------------------------------
    def plot_performance_profiles(self):
        print("📊 4. Generating Dolan-More Performance Profiles...")

        stable_algos = self.df.groupby("algorithm")["is_stabilized"].mean()
        valid_algos = stable_algos[stable_algos > 0.5].index
        subset = self.df[self.df["algorithm"].isin(valid_algos)]

        tasks = subset.groupby(["rho", "A"])

        plt.figure(figsize=(12, 8))
        styles = ["-", "--", "-.", ":"]

        for i, algo in enumerate(valid_algos):
            ratios = []
            algo_data = subset[subset["algorithm"] == algo]

            for (rho, A), _ in tasks:
                best_val = (
                    tasks.get_group((rho, A))
                    .groupby("algorithm")["lyapunov"]
                    .median()
                    .min()
                )
                my_val = algo_data[(algo_data["rho"] == rho) & (algo_data["A"] == A)][
                    "lyapunov"
                ].median()

                if pd.notna(my_val) and best_val > 0:
                    ratios.append(my_val / best_val)
                elif pd.notna(my_val):
                    ratios.append(1e5)

            ratios = np.array(ratios)
            ratios = ratios[ratios < 50]
            if len(ratios) == 0:
                continue

            ratios.sort()
            y = np.arange(1, len(ratios) + 1) / len(tasks)

            plt.step(
                ratios,
                y,
                label=algo,
                linewidth=2,
                where="post",
                linestyle=styles[i % 4],
            )

        plt.xscale("log")
        plt.xlabel(r"Performance Ratio $\tau$ (Log Scale)")
        plt.ylabel(r"Probability of solving within factor $\tau$ of Best")
        plt.title(
            "Fig 4: Dolan-Moré Performance Profile (Reliability)",
            fontsize=16,
            weight="bold",
        )
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

        plt.savefig(self.output_dir / "4_Performance_Profile.png")
        plt.close()

    # -------------------------------------------------------------------------
    # PLOT 5: EFFICIENCY FRONTIER
    # -------------------------------------------------------------------------
    def plot_efficiency_frontier(self):
        print("📊 5. Generating Efficiency Frontier...")

        agg = (
            self.df.groupby(["algorithm", "Type"])
            .agg({"lyapunov": "median", "runtime": "median"})
            .reset_index()
        )

        agg = agg[agg["lyapunov"] < 1e10]

        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            data=agg,
            x="runtime",
            y="lyapunov",
            hue="Type",
            style="Type",
            s=200,
            palette="deep",
        )

        for i, row in agg.iterrows():
            plt.text(
                row["runtime"] * 1.03,
                row["lyapunov"],
                row["algorithm"],
                fontsize=9,
                alpha=0.8,
            )

        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Runtime per seed (seconds)")
        plt.ylabel("Median Lyapunov Error")
        plt.title(
            "Fig 5: Efficiency Frontier (Cost vs Precision)", fontsize=16, weight="bold"
        )
        plt.grid(True, which="both", alpha=0.2)

        plt.arrow(
            agg["runtime"].max(),
            agg["lyapunov"].max(),
            -agg["runtime"].max() * 0.5,
            -agg["lyapunov"].max() * 0.5,
            color="red",
            alpha=0.1,
            width=0.0001,
        )
        plt.text(
            agg["runtime"].min(),
            agg["lyapunov"].min(),
            "OPTIMAL REGION",
            color="green",
            weight="bold",
        )

        plt.savefig(self.output_dir / "5_Efficiency_Frontier.png")
        plt.close()

    # -------------------------------------------------------------------------
    # PLOT 6: SCALING LAW
    # -------------------------------------------------------------------------
    def plot_scaling_law(self):
        print("📊 6. Validating Theoretical Scaling Law...")

        slopes = []
        for (algo, rho), group in self.df.groupby(["algorithm", "rho"]):
            medians = group.groupby("A")["lyapunov"].median().reset_index()
            valid = medians[medians["lyapunov"] < 1e5]

            if len(valid) > 2:
                log_A = np.log10(valid["A"])
                log_V = np.log10(valid["lyapunov"])
                k, _ = np.polyfit(log_A, log_V, 1)

                algo_type = self.df[self.df["algorithm"] == algo]["Type"].iloc[0]
                slopes.append(
                    {
                        "algorithm": algo,
                        "rho": rho,
                        "Measured Slope": k,
                        "Theory": rho + 1.0,
                        "Type": algo_type,
                    }
                )

        res = pd.DataFrame(slopes)
        if res.empty:
            return

        for algo_type in ["First-Order", "Zero-Order"]:
            subset = res[res["Type"] == algo_type]
            if subset.empty:
                continue

            plt.figure(figsize=(14, 7))
            sns.barplot(
                data=subset,
                x="algorithm",
                y="Measured Slope",
                hue="rho",
                palette="viridis",
            )

            colors = sns.color_palette("viridis", 3)
            plt.axhline(
                1.2,
                color=colors[0],
                ls="--",
                alpha=0.8,
                label="Theory ρ=0.2 (Slope=1.2)",
            )
            plt.axhline(
                1.5,
                color=colors[1],
                ls="--",
                alpha=0.8,
                label="Theory ρ=0.5 (Slope=1.5)",
            )
            plt.axhline(
                2.0,
                color=colors[2],
                ls="--",
                alpha=0.8,
                label="Theory ρ=1.0 (Slope=2.0)",
            )

            plt.xticks(rotation=45, ha="right")
            plt.title(
                f"Fig 6 ({algo_type}): Scaling Law Validation ($E[V] \propto A^{{\\rho+1}}$)",
                fontsize=16,
                weight="bold",
            )
            plt.ylim(0, 3.5)
            plt.legend(loc="upper right")

            plt.savefig(self.output_dir / f"6_Scaling_Validation_{algo_type}.png")
            plt.close()

    def run_all(self):
        print("🚀 Starting Visualization (Excluding redundant algos)...\n")
        self.plot_stabilization_rates()
        self.plot_bound_heatmaps()
        self.plot_convergence_profiles()
        self.plot_performance_profiles()
        self.plot_efficiency_frontier()
        self.plot_scaling_law()
        print(f"\n✅ SUCCESS! All figures saved to folder: '{self.output_dir}'")


if __name__ == "__main__":
    viz = ComprehensiveVisualizer("full_results.csv")
    viz.run_all()
