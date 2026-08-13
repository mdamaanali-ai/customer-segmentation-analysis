import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
        }

        .title {
            font-size: 2.4rem;
            font-weight: 700;
            color: #172033;
        }

        .subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: white;
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }

        .metric-title {
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-value {
            color: #172033;
            font-size: 1.8rem;
            font-weight: 700;
        }

        .insight {
            padding: 1rem;
            background: white;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_file(uploaded_file):
    """Read CSV or Excel file."""

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")


def clean_column_names(df):
    """Standardize column names."""

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def convert_numeric_columns(df):
    """Convert columns that contain numeric values."""

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            cleaned = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )

            numeric = pd.to_numeric(
                cleaned,
                errors="coerce"
            )

            # Convert only if most values are numeric
            if numeric.notna().mean() >= 0.70:
                df[column] = numeric

    return df


def find_customer_id(df):
    """Find a likely customer identifier."""

    possible_names = [
        "customer_id",
        "customerid",
        "customer",
        "client_id",
        "user_id",
        "id"
    ]

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def get_numeric_features(df):
    """Return useful numeric columns for clustering."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Remove obvious ID columns
    excluded_words = [
        "id",
        "zip",
        "postal",
        "pincode",
        "phone",
        "mobile"
    ]

    features = []

    for column in numeric_columns:

        if not any(
            word in column.lower()
            for word in excluded_words
        ):
            features.append(column)

    return features


def create_customer_level_data(df, customer_id):
    """
    If multiple rows belong to the same customer,
    aggregate numerical behavior at customer level.
    """

    if customer_id is None:
        return df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        return df.copy()

    customer_data = (
        df.groupby(customer_id)[numeric_columns]
        .agg(["mean", "sum"])
    )

    # Flatten multi-level columns
    customer_data.columns = [
        f"{column}_{aggregation}"
        for column, aggregation in customer_data.columns
    ]

    customer_data.reset_index(inplace=True)

    return customer_data


def create_segment_names(summary):
    """Generate readable names for clusters."""

    names = {}

    if summary.empty:
        return names

    # Rank clusters by average overall score
    summary = summary.copy()

    numeric_columns = summary.select_dtypes(
        include=np.number
    ).columns

    score_columns = [
        column for column in numeric_columns
        if column != "cluster"
    ]

    if not score_columns:
        return names

    summary["segment_score"] = (
        summary[score_columns]
        .rank(pct=True)
        .mean(axis=1)
    )

    ordered = summary.sort_values(
        "segment_score"
    )

    labels = [
        "Low Value Customers",
        "Occasional Customers",
        "Regular Customers",
        "High Value Customers",
        "Premium Customers",
        "VIP Customers"
    ]

    for index, cluster in enumerate(ordered.index):

        if index < len(labels):
            names[cluster] = labels[index]
        else:
            names[cluster] = f"Customer Segment {cluster + 1}"

    return names


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">👥 Customer Segmentation Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover customer groups using behavioral and demographic patterns '
    'with machine learning.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Data Upload")

    uploaded_file = st.file_uploader(
        "Upload customer data",
        type=["csv", "xlsx", "xls"]
    )

    st.divider()

    st.header("⚙️ Clustering Settings")

    number_of_clusters = st.slider(
        "Number of Customer Segments",
        min_value=2,
        max_value=8,
        value=4
    )

    st.caption(
        "K-Means clustering will be used to identify "
        "customers with similar characteristics."
    )


# ============================================================
# NO FILE STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a customer CSV or Excel file from the sidebar "
        "to start the analysis."
    )

    st.subheader("📌 Recommended Dataset")

    sample = pd.DataFrame({
        "customer_id": [1001, 1002, 1003, 1004, 1005],
        "age": [22, 35, 41, 29, 52],
        "annual_income": [
            30000,
            65000,
            85000,
            50000,
            120000
        ],
        "purchase_frequency": [2, 8, 12, 5, 20],
        "total_spending": [
            1200,
            8500,
            15000,
            4200,
            28000
        ]
    })

    st.dataframe(
        sample,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        **Useful columns include:**

        - Customer ID
        - Age
        - Income
        - Purchase Frequency
        - Total Spending
        - Recency
        - Number of Orders
        - Average Order Value
        """
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = load_file(uploaded_file)

    data = clean_column_names(data)

    data = convert_numeric_columns(data)

except Exception as error:

    st.error(f"❌ Could not read the file: {error}")
    st.stop()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

data.dropna(
    axis=0,
    how="all",
    inplace=True
)

data.dropna(
    axis=1,
    how="all",
    inplace=True
)

data.drop_duplicates(
    inplace=True
)


# ============================================================
# CUSTOMER IDENTIFICATION
# ============================================================

customer_id = find_customer_id(data)


if customer_id:

    customer_data = create_customer_level_data(
        data,
        customer_id
    )

else:

    customer_data = data.copy()


# ============================================================
# NUMERIC FEATURE DETECTION
# ============================================================

numeric_features = get_numeric_features(
    customer_data
)


if len(numeric_features) < 2:

    st.error(
        "❌ At least two useful numerical features are required "
        "for customer segmentation."
    )

    st.write("Available columns:")

    st.write(
        customer_data.columns.tolist()
    )

    st.stop()


# ============================================================
# FEATURE SELECTION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("📊 Features")

selected_features = st.sidebar.multiselect(
    "Select clustering features",
    numeric_features,
    default=numeric_features[:min(5, len(numeric_features))]
)


if len(selected_features) < 2:

    st.warning(
        "Please select at least two numerical features."
    )

    st.stop()


# ============================================================
# PREPARE CLUSTERING DATA
# ============================================================

model_data = customer_data[
    selected_features
].copy()

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.fillna(
    model_data.median(numeric_only=True)
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    model_data
)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(
    scaled_features
)


customer_data["cluster"] = clusters


# ============================================================
# SILHOUETTE SCORE
# ============================================================

if len(customer_data) > number_of_clusters:

    silhouette = silhouette_score(
        scaled_features,
        clusters
    )

else:

    silhouette = 0


# ============================================================
# SEGMENT SUMMARY
# ============================================================

segment_summary = (
    customer_data
    .groupby("cluster")[selected_features]
    .mean()
    .round(2)
)


segment_counts = (
    customer_data["cluster"]
    .value_counts()
    .sort_index()
)


segment_summary["customer_count"] = (
    segment_counts
)


# Generate meaningful names
segment_names = create_segment_names(
    segment_summary.drop(
        columns=["customer_count"],
        errors="ignore"
    )
)


customer_data["segment"] = (
    customer_data["cluster"]
    .map(segment_names)
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📊 Segmentation Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                TOTAL CUSTOMERS
            </div>
            <div class="metric-value">
                {len(customer_data):,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                CUSTOMER SEGMENTS
            </div>
            <div class="metric-value">
                {number_of_clusters}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                FEATURES USED
            </div>
            <div class="metric-value">
                {len(selected_features)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                SILHOUETTE SCORE
            </div>
            <div class="metric-value">
                {silhouette:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# CUSTOMER DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader("👥 Customer Segment Distribution")

    distribution = (
        customer_data["segment"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "segment",
        "customers"
    ]

    fig_distribution = px.pie(
        distribution,
        names="segment",
        values="customers",
        hole=0.45,
        title="Customers by Segment"
    )

    fig_distribution.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )


# ============================================================
# SEGMENT SIZE
# ============================================================

with col2:

    st.subheader("📈 Segment Size")

    fig_size = px.bar(
        distribution.sort_values("customers"),
        x="customers",
        y="segment",
        orientation="h",
        text="customers",
        title="Number of Customers per Segment",
        labels={
            "customers": "Customers",
            "segment": "Segment"
        }
    )

    fig_size.update_layout(
        template="plotly_white",
        height=420
    )

    st.plotly_chart(
        fig_size,
        use_container_width=True
    )


# ============================================================
# PCA VISUALIZATION
# ============================================================

st.subheader("🎯 Customer Segments Visualization")

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(
    scaled_features
)

pca_df = pd.DataFrame({
    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Segment": customer_data["segment"].values
})


fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Segment",
    hover_data={
        "PC1": ":.2f",
        "PC2": ":.2f"
    },
    title="Customer Segments Using PCA",
    labels={
        "PC1": "Principal Component 1",
        "PC2": "Principal Component 2"
    }
)

fig_pca.update_layout(
    template="plotly_white",
    height=550
)

st.plotly_chart(
    fig_pca,
    use_container_width=True
)


# ============================================================
# FEATURE ANALYSIS
# ============================================================

st.subheader("🔍 Segment Characteristics")

summary_for_chart = (
    customer_data
    .groupby("segment")[selected_features]
    .mean()
    .reset_index()
)

st.dataframe(
    summary_for_chart,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FEATURE COMPARISON
# ============================================================

feature_to_visualize = st.selectbox(
    "Select a feature to compare across segments",
    selected_features
)


feature_summary = (
    customer_data
    .groupby("segment")[feature_to_visualize]
    .mean()
    .reset_index()
    .sort_values(
        feature_to_visualize,
        ascending=False
    )
)


fig_feature = px.bar(
    feature_summary,
    x="segment",
    y=feature_to_visualize,
    text_auto=".2f",
    title=f"Average {feature_to_visualize} by Customer Segment",
    labels={
        "segment": "Customer Segment",
        feature_to_visualize: feature_to_visualize
    }
)

fig_feature.update_layout(
    template="plotly_white",
    height=450
)

st.plotly_chart(
    fig_feature,
    use_container_width=True
)


# ============================================================
# CUSTOMER SEGMENT INSIGHTS
# ============================================================

st.subheader("💡 Customer Segment Insights")


for cluster, segment_name in segment_names.items():

    segment_data = customer_data[
        customer_data["cluster"] == cluster
    ]

    customer_count = len(segment_data)

    percentage = (
        customer_count /
        len(customer_data)
    ) * 100

    averages = (
        segment_data[selected_features]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    strongest_feature = averages.index[0]

    strongest_value = averages.iloc[0]

    st.markdown(
        f"""
        <div class="insight">

        <h4>👤 {segment_name}</h4>

        <b>Customers:</b> {customer_count:,}
        ({percentage:.1f}% of total customers)

        <br><br>

        <b>Key characteristic:</b>
        {strongest_feature.replace("_", " ").title()}

        <br><br>

        <b>Average value:</b>
        {strongest_value:,.2f}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOMER DATA WITH SEGMENTS
# ============================================================

st.subheader("📋 Customer Segment Data")

display_columns = []

if customer_id:
    display_columns.append(customer_id)

display_columns.extend(selected_features)

display_columns.extend(
    ["cluster", "segment"]
)

display_columns = [
    column
    for column in display_columns
    if column in customer_data.columns
]

st.dataframe(
    customer_data[display_columns],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD RESULT
# ============================================================

csv_data = customer_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Segmented Customer Data",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Segmentation Analysis • "
    "Python • Pandas • Scikit-learn • Plotly • Streamlit"
)