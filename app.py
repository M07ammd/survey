# ==================== TAB 1: Single Prediction ====================
with tab1:
    st.header("Individual Assessment")
    
    col1, col2 = st.columns(2)
    
    # Demographics
    with col1:
        st.subheader("👤 Demographics")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        gender = st.selectbox("Gender", ["m", "f"])
        ethnicity = st.selectbox("Ethnicity", 
            ["White-European", "South Asian", "Black", "Asian", "Latino", 
             "Middle Eastern", "Others", "?"])
        
    with col2:
        st.subheader("🏥 Medical History")
        jaundice = st.selectbox("Jaundice at Birth", ["yes", "no"])
        autism = st.selectbox("Family History of Autism", ["yes", "no"])
        relation = st.selectbox("Relation Type", ["Self", "Parent", "Health care professional", 
                                                   "Relative", "?"])
        country = st.selectbox("Country of Residence", 
            ["United States", "United Kingdom", "Canada", "Australia", "India", "Others"])
        used_app = st.selectbox("Used App Before", ["yes", "no"])
    
    st.subheader("📋 Autism Spectrum Quotient (AQ) Test Questions")
    st.markdown("**Score 1 for each agreement, 0 for each disagreement**")
    
    scores = []
    cols = st.columns(5)
    
    questions = [
        "Q1: Focus on details",
        "Q2: Switch between tasks",
        "Q3: Follow conversation",
        "Q4: Get upset by changes",
        "Q5: Special interests/hobbies",
        "Q6: Notice patterns",
        "Q7: Prefer routines",
        "Q8: Remember factual details",
        "Q9: Find social situations difficult",
        "Q10: Difficulty understanding others"
    ]
    
    for i in range(10):
        with cols[i % 5]:
            score = st.checkbox(f"A{i+1}", key=f"q{i}")
            scores.append(1 if score else 0)
    
    result_score = st.number_input("AQ Test Score (0-10)", min_value=0, max_value=10, value=5)
    
    # Prediction Button
    if st.button("🎯 Get Prediction", use_container_width=True, type="primary"):
        try:
            # ================= INPUT DATA =================
            input_data = pd.DataFrame({
                'A1_Score': [scores[0]],
                'A2_Score': [scores[1]],
                'A3_Score': [scores[2]],
                'A4_Score': [scores[3]],
                'A5_Score': [scores[4]],
                'A6_Score': [scores[5]],
                'A7_Score': [scores[6]],
                'A8_Score': [scores[7]],
                'A9_Score': [scores[8]],
                'A10_Score': [scores[9]],
                'age': [age],
                'gender': [gender],
                'ethnicity': [ethnicity if ethnicity != "?" else "White-European"],
                'jaundice': [jaundice],
                'autism': [autism],
                'Country_of_res': [country if country != "Others" else "United States"],
                'used_app_before': [used_app],
                'result': [result_score],
                'relation': [relation if relation != "?" else "Self"]
            })
            
            # ================= FIX ONLY (NO REMOVAL) =================
            for col in input_data.columns:
                if col in encoders:
                    try:
                        input_data[col] = encoders[col].transform(
                            input_data[col].astype(str)
                        )
                    except:
                        input_data[col] = 0
            
            # IMPORTANT: keep feature alignment
            input_data = input_data.reindex(columns=feature_cols, fill_value=0)
            
            # ================= PREDICTION =================
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            autism_prob = probability[1] * 100
            non_autism_prob = probability[0] * 100
            
            # ================= DISPLAY =================
            st.markdown("---")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="autism-high">
                        <h2>⚠️ Likely Autistic Traits Detected</h2>
                        <h3 style="font-size: 2.5rem; margin: 10px 0;">{autism_prob:.1f}%</h3>
                        <p style="font-size: 1.1rem;">Risk of Autism Spectrum Disorder</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="autism-low">
                        <h2>✅ Low Risk Assessment</h2>
                        <h3 style="font-size: 2.5rem; margin: 10px 0;">{autism_prob:.1f}%</h3>
                        <p style="font-size: 1.1rem;">Risk of Autism Spectrum Disorder</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ================= METRICS =================
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Autism Probability", f"{autism_prob:.2f}%")
            with col2:
                st.metric("Non-Autism Probability", f"{non_autism_prob:.2f}%")
            
            # ================= GAUGE (UNCHANGED) =================
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=autism_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Autism Risk Level"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "#90EE90"},
                        {'range': [25, 50], 'color': "#FFD700"},
                        {'range': [50, 75], 'color': "#FFA500"},
                        {'range': [75, 100], 'color': "#FF6347"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=400, width=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
