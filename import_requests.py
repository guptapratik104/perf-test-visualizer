import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

API_KEY = os.environ.get('NEW_RELIC_API_KEY', '')
ACCOUNT_ID = os.environ.get('NEW_RELIC_ACCOUNT_ID', '')

if not API_KEY:
    st.warning("⚠️ Please set the NEW_RELIC_API_KEY environment variable")
    
if not ACCOUNT_ID:
    st.warning("⚠️ Please set the NEW_RELIC_ACCOUNT_ID environment variable")

NRQL_QUERY = '''
FROM Transaction 
SELECT
  average(duration) AS avg_duration,
  percentile(duration, 95) AS p95_duration,
  rate(count(*), 1 minute) AS throughput,
  (filter(count(*), WHERE error IS TRUE) / count(*)) * 100 AS error_rate
FACET appName
SINCE 30 minutes ago LIMIT 80
'''

def fetch_nr_data(api_key, account_id, nrql):
    headers = {'API-Key': api_key}
    graphql_query = '''
    {
      actor {
        account(id: %s) {
          nrql(query: """%s""") {
            results
          }
        }
      }
    }
    ''' % (account_id, nrql)

    try:
        response = requests.post(
            'https://api.newrelic.com/graphql',
            json={'query': graphql_query},
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Check if response has the expected structure
        if not data or 'data' not in data:
            st.error("Invalid API response format")
            return []
            
        if 'actor' not in data['data'] or data['data']['actor'] is None:
            st.error("No actor data in API response")
            return []
            
        if 'account' not in data['data']['actor'] or data['data']['actor']['account'] is None:
            st.error("No account data in API response")
            return []
            
        if 'nrql' not in data['data']['actor']['account'] or data['data']['actor']['account']['nrql'] is None:
            st.error("No NRQL data in API response")
            return []
            
        if 'results' not in data['data']['actor']['account']['nrql'] or data['data']['actor']['account']['nrql']['results'] is None:
            st.error("No results in API response")
            return []
            
        return data['data']['actor']['account']['nrql']['results']
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return []
    except ValueError as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return []

def calculate_apdex_score(response_time_ms, threshold_ms=500):
    """
    Calculate Apdex score based on response time
    Apdex = (Satisfied + Tolerating/2) / Total
    - Satisfied: <= threshold
    - Tolerating: <= 4 * threshold  
    - Frustrated: > 4 * threshold
    """
    if response_time_ms <= threshold_ms:
        return 100 
    elif response_time_ms <= 4 * threshold_ms:
        return 50   
    else:
        return 0    

def calculate_error_rate_score(error_rate):
    """
    Industry standard error rate scoring
    - 0%: Perfect (100)
    - 0-1%: Excellent (90-99)
    - 1-5%: Good (70-89)
    - 5-10%: Warning (40-69)
    - >10%: Critical (0-39)
    """
    if error_rate == 0:
        return 100
    elif error_rate <= 1:
        return 90 + (1 - error_rate) * 10  # 90-100
    elif error_rate <= 5:
        return 70 + (5 - error_rate) * 5   # 70-90
    elif error_rate <= 10:
        return 40 + (10 - error_rate) * 6  # 40-70
    else:
        return max(0, 40 - (error_rate - 10) * 4)  # 0-40

def calculate_throughput_score(throughput, median_throughput):
    """
    Throughput scoring based on percentile ranking
    Uses logarithmic scaling to handle wide throughput ranges
    """
    if throughput <= 0:
        return 0
    
    if median_throughput <= 0:
        return 50  # Neutral score if no baseline
    
    # Use log ratio to handle exponential differences
    ratio = throughput / median_throughput
    log_ratio = np.log2(max(ratio, 0.01))  # Prevent log(0)
    
    # Score between 0-100 with 50 at median
    score = 50 + (log_ratio * 15)  # 15 points per doubling
    return np.clip(score, 0, 100)

def calculate_composite_score(throughput_score, latency_score, error_rate_score, 
                            throughput, error_rate):
    """
    Industry-standard weighted composite scoring
    Weights adjust based on service characteristics
    """
    
    # Base weights
    throughput_weight = 0.3
    latency_weight = 0.4
    error_weight = 0.3
    
    # Dynamic weight adjustment based on service profile
    
    # High-throughput services: prioritize throughput more
    if throughput > 1000:  # req/min
        throughput_weight = 0.4
        latency_weight = 0.35
        error_weight = 0.25
    
    # Low-throughput services: prioritize latency and reliability
    elif throughput < 10:
        throughput_weight = 0.2
        latency_weight = 0.4
        error_weight = 0.4
    
    # Error-prone services: heavily penalize errors
    if error_rate > 5:
        error_weight = 0.5
        throughput_weight = 0.25
        latency_weight = 0.25
    
    # Calculate weighted score
    composite = (throughput_score * throughput_weight + 
                latency_score * latency_weight + 
                error_rate_score * error_weight)
    
    # Apply error rate penalty (industry standard)
    if error_rate > 10:
        composite *= 0.5  # 50% penalty for critical error rates
    elif error_rate > 5:
        composite *= 0.8  # 20% penalty for high error rates
    
    return composite

def calculate_performance_scores(df):
    """
    Calculate comprehensive performance scores using industry standards
    """
    # Calculate median throughput for baseline
    median_throughput = df['Throughput'].median()
    
    # Calculate individual scores
    df['Throughput Score'] = df['Throughput'].apply(
        lambda x: calculate_throughput_score(x, median_throughput)
    )
    
    df['Latency Score'] = df['P95 Duration'].apply(
        lambda x: calculate_apdex_score(x, threshold_ms=100)  # 100ms threshold
    )
    
    df['Error Rate Score'] = df['Error Rate'].apply(calculate_error_rate_score)
    
    # Calculate composite score with dynamic weighting
    df['Overall Score'] = df.apply(
        lambda row: calculate_composite_score(
            row['Throughput Score'], 
            row['Latency Score'], 
            row['Error Rate Score'],
            row['Throughput'],
            row['Error Rate']
        ), axis=1
    )
    
    return df

def get_performance_grade(score):
    """Convert numeric score to letter grade"""
    if score >= 90:
        return "A+ (Excellent)", "🟢"
    elif score >= 80:
        return "A (Very Good)", "🟢"
    elif score >= 70:
        return "B (Good)", "🟡"
    elif score >= 60:
        return "C (Fair)", "🟡"
    elif score >= 50:
        return "D (Poor)", "🟠"
    else:
        return "F (Critical)", "🔴"

def main():
    st.title('🚦 Service Performance Scoring Dashboard')
    st.markdown("*Using industry-standard Apdex methodology and weighted scoring*")

    data = fetch_nr_data(API_KEY, ACCOUNT_ID, NRQL_QUERY)
    if not data:
        st.error("No data available")
        return
    
    rows = []
    for item in data:
        # Convert durations from seconds to milliseconds for consistency
        avg_duration_ms = (item.get('avg_duration') or 0) * 1000
        p95_duration_raw = item.get('p95_duration')
        
        # Handle p95_duration - it might be a dict or direct value
        if isinstance(p95_duration_raw, dict):
            p95_duration_ms = (p95_duration_raw.get('95') or 0) * 1000
        else:
            p95_duration_ms = (p95_duration_raw or 0) * 1000
        
        rows.append({
            'Service': item.get('facet'),
            'Average Duration': avg_duration_ms,  # Now in milliseconds
            'P95 Duration': p95_duration_ms,      # Now in milliseconds
            'Throughput': item.get('throughput') or 0,  # req/min
            'Error Rate': item.get('error_rate') or 0,  # percentage
        })

    df = pd.DataFrame(rows)

    if df.empty:
        st.error("No data available for the selected query.")
        return

    # Clean and convert data
    numeric_cols = ['Average Duration', 'P95 Duration', 'Throughput', 'Error Rate']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    # Calculate performance scores
    df = calculate_performance_scores(df)
    df_sorted = df.sort_values(by='Overall Score', ascending=False)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Services", len(df))
    with col2:
        avg_score = df['Overall Score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}/100")
    with col3:
        excellent_services = len(df[df['Overall Score'] >= 90])
        st.metric("Excellent Services", f"{excellent_services}")
    with col4:
        critical_services = len(df[df['Overall Score'] < 50])
        st.metric("Critical Services", f"{critical_services}")

    # Service Performance Table
    st.subheader("📊 Service Performance Ranking")
    
    # Prepare display dataframe
    display_df = df_sorted[['Service', 'Overall Score', 'Throughput', 'P95 Duration', 'Error Rate']].copy()
    display_df['Grade'] = display_df['Overall Score'].apply(lambda x: get_performance_grade(x)[0])
    display_df['Status'] = display_df['Overall Score'].apply(lambda x: get_performance_grade(x)[1])
    
    # Display the table with proper formatting
    st.dataframe(
        display_df,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Performance status",
                width="small",
            ),
            "Overall Score": st.column_config.NumberColumn(
                "Overall Score",
                help="Overall performance score (0-100)",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f",
            ),
            "Throughput": st.column_config.NumberColumn(
                "Throughput",
                help="Requests per minute",
                format="%.1f req/min",
            ),
            "P95 Duration": st.column_config.NumberColumn(
                "P95 Duration", 
                help="95th percentile response time",
                format="%.1f ms",
            ),
            "Error Rate": st.column_config.NumberColumn(
                "Error Rate",
                help="Percentage of failed requests", 
                format="%.2f%%",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Performance distribution chart
    st.subheader("📈 Performance Distribution")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2E8B57' if score >= 70 else '#FFD700' if score >= 50 else '#DC143C' 
              for score in df_sorted['Overall Score']]
    
    bars = ax.barh(df_sorted['Service'], df_sorted['Overall Score'], color=colors)
    ax.set_xlabel('Overall Performance Score (0-100)')
    ax.set_title('Service Performance Ranking\n(Green: Good, Yellow: Fair, Red: Critical)')
    ax.set_xlim(0, 100)
    
    # Add score labels on bars
    for i, (bar, score) in enumerate(zip(bars, df_sorted['Overall Score'])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', va='center', fontsize=9)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()