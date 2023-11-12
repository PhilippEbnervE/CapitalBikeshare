import streamlit as st
import pandas as pd
import plotly.express as px


def recommendations_app():
	# Title and Introduction
	st.title('Revolutionizing Urban Mobility: Capital Bikeshare’s Future with AI')

	st.write("Welcome to the future of urban transportation. Discover how AI-driven demand prediction can transform Capital Bikeshare.")
	st.write("""
	Our AI consultancy has developed a sophisticated demand prediction model tailored for both registered and casual users of Capital Bikeshare. This model leverages advanced algorithms and real-time data analytics to forecast bike usage patterns with high accuracy. The recommended steps to take on the basis of this model can be foudn below.
	""")

	st.header("Business Case Applications")
	application = st.selectbox("Select Recommendation", ['Bike and Dock Availability', 'Marketing and Membership Drives', 'Strategic Planning and Expansion'])

	if application == 'Bike and Dock Availability':
		st.subheader("Optimizing Bike and Dock Availability")
		st.write('- **Application:** The model predicts high-demand times, allowing for strategic bike allocation and optimization of redistribution services by 3rd party contractor.\n\n'
			'- **Benefits:**\n\n'
	        	'*Increased User Satisfaction:* Ensures bikes and docks are available when and where needed, enhancing customer experience.\n\n'
	        	'*Operational Efficiency:* Reduces costs associated with moving bikes and managing stations.\n\n'
	        '- **Implementation:** Deploy smart logistics that respond to predictive data, ensuring optimal allocation of bikes across stations."""\n\n')

	elif application == 'Marketing and Membership Drives':
		st.subheader("Targeted Marketing and Membership Drives")
		st.write('- **Application:** Utilize demand trends to identify potential areas and demographics for marketing. Hone in on commuter benefits and time marketing push for casual riders during good weather and high demand seasons\n\n'
	        	'- **Benefits:**\n\n'
	        	'*Growth in Membership:* Identifies untapped markets and peak interest times for promotional campaigns.\n\n'
	        	'*Enhanced ROI on Marketing Spend:* Focuses marketing efforts on high-potential user groups and areas.\n\n'
	        	'- **Implementation:** Collaborate with marketing teams to launch campaigns aligned with predictive insights, focusing on times and areas of projected high demand.')

	elif application == 'Strategic Planning and Expansion':
	    st.subheader("Strategic Planning and Expansion")
	    st.write('- **Application:** Provides data-driven insights for future station placements and service expansions.Idenfity commuter routes and routes that casual riders tend to take\n\n'
	        '- **Benefits:**\n\n'
	        	'*Informed Expansion Decisions:* Pinpoints locations with likely high usage, avoiding underused stations.\n\n'
	        	'*Long-Term Growth Strategy:* Aligns expansion plans with actual user demand trends.\n\n'
	        '- **Implementation:** Use predictive data to guide infrastructure investment, ensuring new stations are placed in high-demand, underserved areas.')

	# Footer
	st.write("© 2023 Capital Bikeshare AI Consultancy")