A dashboard that ranks your services using New Relic metrics and industry-standard performance scoring. This tool automatically evaluates and grades your services based on throughput, latency, and error rates. It applies Apdex methodology with intelligent scoring to help quickly identify which services need attention.


Key Features
	Real-time metrics from New Relic
	A-F grading system with color-coding
	Dynamic scoring weights that adapt to service characteristics
	Performance breakdown with detailed metric analysis
	Visual distribution chart for at-a-glance comparison

🚀 Quick Start

**1. Set environment variables:**
   
   export NEW_RELIC_API_KEY="your_key"
	 export NEW_RELIC_ACCOUNT_ID="your_id"

**2. Install dependencies:**

	 pip install streamlit requests pandas matplotlib numpy python-dotenv

**3. Run the dashboard:**

	streamlit run import_requests.py
