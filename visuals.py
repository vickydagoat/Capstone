from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def _finish(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def seasonal_cycle(df):
    m = df.groupby(["month", "city"])["tmax"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    m.plot(ax=ax, marker="o")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("Mean Tmax (°C)")
    ax.set_title("Seasonal cycle of daily maximum temperature, 2000-2024")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="City")
    _finish(fig, "01_seasonal_cycle")


def city_month_heatmap(df):
    p = df.pivot_table(index="city", columns="month", values="tmax", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(p, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean Tmax (°C)"})
    ax.set_title("Mean Tmax by city and month")
    _finish(fig, "02_city_month_heatmap")


def hot_days_per_year(df, threshold=40):
    h = df.assign(hot=df["tmax"] >= threshold).groupby(["year", "city"])["hot"].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    h.plot(ax=ax, marker=".")
    ax.set_ylabel(f"Days with Tmax >= {threshold}°C"); ax.set_title("Hot days per year")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="City")
    _finish(fig, "03_hot_days_per_year")


def tmax_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="city", y="tmax", ax=ax)
    ax.axhline(40, color="red", linestyle="--", label="40°C")
    ax.set_ylabel("Tmax (°C)"); ax.set_title("Distribution of daily Tmax by city"); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30)
    _finish(fig, "04_tmax_distribution")


def anomaly_timeline(df, city="Delhi"):
    d = df[df["city"] == city]
    year = d.groupby("year")["heatwave"].sum().idxmax()      # worst year for this city
    d = d[(d["year"] == year) & d["month"].between(3, 7)]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(d["date"], d["tmax"], label="Tmax")
    ax.plot(d["date"], d["tnormal"], "--", label="Tnormal (climatology)")
    ax.fill_between(d["date"], d["tnormal"], d["tmax"], where=(d["heatwave"] == 1).values,
                    color="red", alpha=0.35, label="Heatwave day")
    ax.set_ylabel("°C"); ax.set_title(f"{city} {year}: Tmax vs normal (Mar-Jul)"); ax.legend()
    _finish(fig, "05_anomaly_timeline")


def severity_by_city(df):
    c = df[df["heatwave"] == 1].groupby(["city", "severity_label"]).size().unstack(fill_value=0)
    c = c.reindex(sorted(df["city"].unique()), fill_value=0).reindex(columns=["Moderate", "Severe"], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    c.plot(kind="bar", stacked=True, color=["orange", "darkred"], ax=ax)
    ax.set_ylabel("Heatwave days (2000-2024)"); ax.set_title("Heatwave days by city and severity")
    plt.setp(ax.get_xticklabels(), rotation=30)
    _finish(fig, "06_severity_by_city")


def class_balance(df):
    c = df["severity_label"].value_counts().reindex(["Normal", "Moderate", "Severe"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    c.plot(kind="bar", color=["steelblue", "orange", "darkred"], ax=ax, logy=True)
    ax.set_ylabel("Days (log scale)"); ax.set_title("Class imbalance of severity labels")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _finish(fig, "07_class_balance")


def city_map(df):
    s = df.groupby("city").agg(lat=("lat", "first"), lon=("lon", "first"), hw=("heatwave", "sum"))
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(s["lon"], s["lat"], s=80 + 4 * s["hw"], c=s["hw"], cmap="YlOrRd", edgecolor="k")
    for city, r in s.iterrows():
        ax.annotate(city, (r["lon"], r["lat"]), xytext=(6, 6), textcoords="offset points")
    plt.colorbar(sc, label="Heatwave days 2000-2024")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Heatwave exposure by city")
    _finish(fig, "08_city_map")
