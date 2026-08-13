# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# Using Python, Pandas, Scikit-Learn, Plotly & Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
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
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172554;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .insight-box {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #2563eb;
        margin-bottom: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAMPLE DATA GENERATOR
# ============================================================

@st.cache_data
def generate_sample_data(n=500):

    np.random.seed(42)

    customer_ids = [
        f"CUST-{str(i).zfill(4)}"
        for i in range(1, n + 1)
    ]

    age = np.random.randint(18, 70, n)

    income = np.random.randint(
        20000,
        150000,
        n
    )

    purchase_frequency = np.random.randint(
        1,
        25,
        n
    )

    average_order_value = np.random.randint(
        500,
        15000,
        n
    )

    total_purchases = (
        purchase_frequency *
        average_order_value
    )

    website_visits = np.random.randint(
        1,
        50,
        n
    )

    product_categories = np.random.choice(
        [
            "Electronics",
            "Fashion",
            "Grocery",
            "Beauty",
            "Sports"
        ],
        n
    )

    satisfaction = np.random.randint(
        1,
        11,
        n
    )

    data = pd.DataFrame({

        "Customer_ID": customer_ids,

        "Age": age,

        "Annual_Income": income,

        "Purchase_Frequency": purchase_frequency,

        "Average_Order_Value": average_order_value,

        "Total_Purchases": total_purchases,

        "Website_Visits": website_visits,

        "Product_Category": product_categories,

        "Satisfaction_Score": satisfaction
    })

    return data


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_uploaded_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    if file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file format. Please upload CSV or Excel."
    )


# ============================================================
# FIND NUMERIC COLUMNS
# ============================================================

def get_numeric_columns(data):

    return data.select_dtypes(
        include=np.number
    ).columns.tolist()


# ============================================================
# CREATE CUSTOMER SEGMENTS
# ============================================================

def perform_clustering(data, features, number_of_clusters):

    working_data = data[features].copy()

    # Convert everything to numeric
    for column in working_data.columns:

        working_data[column] = pd.to_numeric(
            working_data[column],
            errors="coerce"
        )

    # Fill missing values
    working_data = working_data.fillna(
        working_data.median(numeric_only=True)
    )

    # Standardization
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        working_data
    )

    # K-Means
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(
        scaled_data
    )

    result = data.copy()

    result["Segment"] = clusters + 1

    return (
        result,
        scaled_data,
        model,
        scaler
    )


# ============================================================
# SEGMENT NAMES
# ============================================================

def assign_segment_names(summary):

    names = {}

    if "Total_Purchases" in summary.columns:

        sorted_segments = (
            summary["Total_Purchases"]
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        labels = [
            "High Value Customers",
            "Potential Customers",
            "Regular Customers",
            "Low Value Customers",
            "Occasional Customers"
        ]

        for i, segment in enumerate(sorted_segments):

            if i < len(labels):
                names[segment] = labels[i]

    else:

        for segment in summary.index:

            names[segment] = (
                f"Customer Segment {segment}"
            )

    return names


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown(
    "### Data Source"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv", "xlsx", "xls"]
)


if uploaded_file is not None:

    try:

        data = load_uploaded_file(
            uploaded_file
        )

        st.sidebar.success(
            "Dataset uploaded successfully!"
        )

    except Exception as error:

        st.sidebar.error(
            f"Unable to read file: {error}"
        )

        data = generate_sample_data()

else:

    st.sidebar.info(
        "No dataset uploaded. "
        "Using sample customer data."
    )

    data = generate_sample_data()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">👥 Customer Segmentation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyze customer behavior, purchasing patterns and demographics
    using K-Means clustering.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA CLEANING
# ============================================================

data = data.copy()

data = data.drop_duplicates()

data.columns = [
    str(column).strip()
    for column in data.columns
]


# ============================================================
# BASIC INFORMATION
# ============================================================

numeric_columns = get_numeric_columns(data)

if len(numeric_columns) < 2:

    st.error(
        "The dataset must contain at least two numerical columns "
        "for customer segmentation."
    )

    st.stop()


# ============================================================
# SIDEBAR FEATURE SELECTION
# ============================================================

st.sidebar.markdown(
    "### Clustering Features"
)

default_features = [
    column
    for column in numeric_columns
    if column.lower() in [
        "age",
        "annual_income",
        "income",
        "purchase_frequency",
        "average_order_value",
        "total_purchases",
        "spending_score",
        "website_visits",
        "satisfaction_score"
    ]
]

if len(default_features) < 2:

    default_features = numeric_columns[:min(5, len(numeric_columns))]


features = st.sidebar.multiselect(
    "Select features for clustering",
    options=numeric_columns,
    default=default_features
)


if len(features) < 2:

    st.warning(
        "Please select at least two numerical features."
    )

    st.stop()


# ============================================================
# NUMBER OF CLUSTERS
# ============================================================

number_of_clusters = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=2,
    max_value=8,
    value=4,
    step=1
)


# ============================================================
# PERFORM CLUSTERING
# ============================================================

try:

    clustered_data, scaled_data, model, scaler = (
        perform_clustering(
            data,
            features,
            number_of_clusters
        )
    )

except Exception as error:

    st.error(
        f"Clustering failed: {error}"
    )

    st.stop()


# ============================================================
# SILHOUETTE SCORE
# ============================================================

try:

    silhouette = silhouette_score(
        scaled_data,
        clustered_data["Segment"]
    )

except Exception:

    silhouette = 0


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Business Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Customers",
        f"{len(clustered_data):,}"
    )


with col2:

    st.metric(
        "Customer Segments",
        number_of_clusters
    )


with col3:

    st.metric(
        "Features Used",
        len(features)
    )


with col4:

    st.metric(
        "Silhouette Score",
        f"{silhouette:.3f}"
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Customer Data</div>',
    unsafe_allow_html=True
)

with st.expander(
    "View Dataset"
):

    st.dataframe(
        clustered_data,
        use_container_width=True,
        height=350
    )


# ============================================================
# SEGMENT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📌 Customer Segment Summary</div>',
    unsafe_allow_html=True
)

summary = (
    clustered_data
    .groupby("Segment")[features]
    .mean()
    .round(2)
)

summary["Customer_Count"] = (
    clustered_data
    .groupby("Segment")
    .size()
)

summary["Customer_Percentage"] = (
    summary["Customer_Count"] /
    len(clustered_data) *
    100
).round(2)


segment_names = assign_segment_names(
    summary
)

summary["Segment_Name"] = [
    segment_names.get(
        segment,
        f"Segment {segment}"
    )
    for segment in summary.index
]


display_columns = [
    "Segment_Name",
    "Customer_Count",
    "Customer_Percentage"
] + features


st.dataframe(
    summary[display_columns],
    use_container_width=True
)


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    segment_counts = (
        clustered_data["Segment"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    segment_counts["Segment_Name"] = (
        segment_counts["Segment"]
        .map(segment_names)
    )

    fig_pie = px.pie(
        segment_counts,
        names="Segment_Name",
        values="Customers",
        hole=0.45,
        title="Customer Segment Distribution"
    )

    fig_pie.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# ============================================================
# SEGMENT SIZE BAR CHART
# ============================================================

with col2:

    fig_bar = px.bar(
        segment_counts,
        x="Segment_Name",
        y="Customers",
        text="Customers",
        title="Number of Customers in Each Segment"
    )

    fig_bar.update_traces(
        textposition="outside"
    )

    fig_bar.update_layout(
        height=450,
        xaxis_title="Customer Segment",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ============================================================
# PCA VISUALIZATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Customer Segmentation Visualization</div>',
    unsafe_allow_html=True
)

pca = PCA(
    n_components=2,
    random_state=42
)

pca_data = pca.fit_transform(
    scaled_data
)

clustered_data["PCA_1"] = pca_data[:, 0]

clustered_data["PCA_2"] = pca_data[:, 1]

clustered_data["Segment_Name"] = (
    clustered_data["Segment"]
    .map(segment_names)
)


fig_scatter = px.scatter(
    clustered_data,
    x="PCA_1",
    y="PCA_2",
    color="Segment_Name",
    hover_data=features,
    title="Customer Segments using PCA",
    labels={
        "PCA_1": "Principal Component 1",
        "PCA_2": "Principal Component 2",
        "Segment_Name": "Customer Segment"
    }
)

fig_scatter.update_layout(
    height=600
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ============================================================
# FEATURE COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">📈 Segment Characteristics</div>',
    unsafe_allow_html=True
)

selected_feature = st.selectbox(
    "Choose a feature to compare",
    features
)


fig_feature = px.box(
    clustered_data,
    x="Segment_Name",
    y=selected_feature,
    color="Segment_Name",
    points="all",
    title=f"{selected_feature} Across Customer Segments"
)

fig_feature.update_layout(
    height=500,
    showlegend=False
)

st.plotly_chart(
    fig_feature,
    use_container_width=True
)


# ============================================================
# SEGMENT PROFILE HEATMAP
# ============================================================

st.markdown(
    '<div class="section-title">🔥 Segment Profile</div>',
    unsafe_allow_html=True
)

heatmap_data = (
    summary[features]
    .copy()
)

# Normalize each feature between 0 and 1
for column in heatmap_data.columns:

    minimum = heatmap_data[column].min()

    maximum = heatmap_data[column].max()

    if maximum != minimum:

        heatmap_data[column] = (
            heatmap_data[column] - minimum
        ) / (
            maximum - minimum
        )

    else:

        heatmap_data[column] = 0


heatmap_data.index = [
    segment_names.get(
        segment,
        f"Segment {segment}"
    )
    for segment in heatmap_data.index
]


fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text=np.round(
            heatmap_data.values,
            2
        ),
        texttemplate="%{text}",
        hovertemplate=(
            "Segment: %{y}<br>"
            "Feature: %{x}<br>"
            "Relative Score: %{z:.2f}"
            "<extra></extra>"
        )
    )
)

fig_heatmap.update_layout(
    title="Relative Segment Characteristics",
    height=450
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)


# ============================================================
# AUTOMATIC BUSINESS INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Business Insights</div>',
    unsafe_allow_html=True
)


def get_feature_value(segment, feature):

    try:

        return summary.loc[
            segment,
            feature
        ]

    except Exception:

        return None


for segment in summary.index:

    segment_name = segment_names.get(
        segment,
        f"Segment {segment}"
    )

    customer_count = summary.loc[
        segment,
        "Customer_Count"
    ]

    percentage = summary.loc[
        segment,
        "Customer_Percentage"
    ]

    insight_parts = []

    if "Annual_Income" in features:

        income = get_feature_value(
            segment,
            "Annual_Income"
        )

        if income is not None:

            insight_parts.append(
                f"average annual income of ₹{income:,.0f}"
            )

    elif "Income" in features:

        income = get_feature_value(
            segment,
            "Income"
        )

        if income is not None:

            insight_parts.append(
                f"average income of ₹{income:,.0f}"
            )


    if "Purchase_Frequency" in features:

        frequency = get_feature_value(
            segment,
            "Purchase_Frequency"
        )

        if frequency is not None:

            insight_parts.append(
                f"purchase frequency of {frequency:.1f}"
            )


    if "Average_Order_Value" in features:

        order_value = get_feature_value(
            segment,
            "Average_Order_Value"
        )

        if order_value is not None:

            insight_parts.append(
                f"average order value of ₹{order_value:,.0f}"
            )


    if "Total_Purchases" in features:

        purchases = get_feature_value(
            segment,
            "Total_Purchases"
        )

        if purchases is not None:

            insight_parts.append(
                f"total purchases averaging ₹{purchases:,.0f}"
            )


    if insight_parts:

        description = ", ".join(
            insight_parts
        )

    else:

        description = (
            "distinct purchasing and demographic characteristics"
        )


    st.markdown(
        f"""
        <div class="insight-box">

        <strong>👤 {segment_name}</strong>

        <br><br>

        This segment contains
        <strong>{customer_count:,} customers</strong>
        ({percentage:.1f}% of the customer base).

        The segment shows {description}.

        <br><br>

        <strong>Recommended Strategy:</strong>

        """

        + (

            "Focus on premium products, loyalty rewards and personalized offers."
            if segment == summary[
                "Total_Purchases"
            ].idxmax()
            and "Total_Purchases" in summary.columns

            else

            "Use personalized promotions and targeted campaigns to increase engagement."

        )

        + """

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOMER PREFERENCE ANALYSIS
# ============================================================

categorical_columns = data.select_dtypes(
    exclude=np.number
).columns.tolist()


if categorical_columns:

    st.markdown(
        '<div class="section-title">🛍️ Customer Preferences</div>',
        unsafe_allow_html=True
    )

    category_column = st.selectbox(
        "Select a categorical feature",
        categorical_columns
    )

    preference_data = (
        clustered_data
        .groupby(
            ["Segment_Name", category_column]
        )
        .size()
        .reset_index(
            name="Customers"
        )
    )

    fig_preferences = px.bar(
        preference_data,
        x=category_column,
        y="Customers",
        color="Segment_Name",
        barmode="group",
        title=f"{category_column} Preferences by Customer Segment"
    )

    fig_preferences.update_layout(
        height=550
    )

    st.plotly_chart(
        fig_preferences,
        use_container_width=True
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Marketing Recommendations</div>',
    unsafe_allow_html=True
)

recommendation_col1, recommendation_col2 = st.columns(2)


with recommendation_col1:

    st.markdown(
        """
        <div class="insight-box">

        <strong>💎 High-Value Customers</strong>

        <br><br>

        • Provide loyalty rewards<br>
        • Offer premium products<br>
        • Give early access to new products<br>
        • Use personalized recommendations

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="insight-box">

        <strong>📈 Potential Customers</strong>

        <br><br>

        • Use targeted discounts<br>
        • Recommend related products<br>
        • Encourage repeat purchases<br>
        • Use email and digital campaigns

        </div>
        """,
        unsafe_allow_html=True
    )


with recommendation_col2:

    st.markdown(
        """
        <div class="insight-box">

        <strong>🛒 Regular Customers</strong>

        <br><br>

        • Create loyalty programs<br>
        • Offer bundle discounts<br>
        • Encourage higher order values<br>
        • Recommend complementary products

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="insight-box">

        <strong>🌱 Low-Engagement Customers</strong>

        <br><br>

        • Send re-engagement campaigns<br>
        • Provide limited-time offers<br>
        • Understand reasons for low activity<br>
        • Recommend popular products

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DOWNLOAD SEGMENTED DATA
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Export Results</div>',
    unsafe_allow_html=True
)

download_data = clustered_data.copy()

csv_data = download_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Segmented Customer Data",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <hr>

    <center>

    <b>Customer Segmentation Project</b><br>

    Built with Python • Pandas • Scikit-Learn • Plotly • Streamlit

    </center>

    """,
    unsafe_allow_html=True
)