import time

import streamlit as st

st.set_page_config(layout='wide',page_title='MatFlow', initial_sidebar_state='collapsed')

from subpage.navbar import navbar,vspace
import streamlit.components.v1 as components
selected2 = navbar()
vspace(7)


# if selected2 == "Contact":
#     switch_page('contact')
# if selected2 == "Demo":
#     switch_page('demo')


with st.container():
    img1, img2, img3 = st.columns((4, 25, 1))
    with img2:
        st.image(
            'https://i.ibb.co/SmxY6J9/banner.png')

    c0, tit2,c1 = st.columns([.5, 20,.5])
    with tit2:
        st.markdown(
            '''
            <div class="" style="margin-left: 100px; margin-right: 100px">
                <div class='d-flex justify-content-center'>
                    <h2 style='font-size: 35px;font-weight: 100%; color:purple; text-transform: uppercase; text-align: center';>
                    A Machine Learning Based Data Analysis and Exploration System For Material Design</h2>
                </div>
        ''', unsafe_allow_html=True
        )
        st.markdown("""
                         <div
                           class="des-title" style='text-align: justify; font-weight: bold; 
                           background-color: #fbffe3; color: #0aa0c2;font-family: Tahoma, 
                           Verdana, sans-serif; font-size: 20px;  text-align: center'>
                           MatFlow is a web-based dataflow framework for visual data exploration. A machine 
                           learning-based data analysis and exploration system for material design is a computer 
                           system that uses machine learning algorithms to analyze and explore large amounts of 
                           material design data. This system can be used to identify patterns and relationships 
                           in the data, generate insights and predictions, and support decision-making in the field 
                           of material design. The system can be trained on existing data to improve its accuracy 
                           and can also be updated with new data as it becomes available to continue learning and 
                           improving its performance.
                         </div>    
        
                         <br>
                         <h3 class="title" style='text-align: center; text-decoration: overline underline; 
                         background-color:#fbffe3; font-family: "Lucida Console", "Courier New", monospace; 
                         font-weight: bold; color:purple'>
                         <u>Our Next Goal</u></h3>
                          <div style='text-align: left; font-weight: bold; 
                           background-color: #fbffe3; color: #0aa0c2;font-family: Tahoma, 
                           Verdana, sans-serif; font-size: 20px;  text-align: center'>
                           A cloud-based smart novel materials discovery system using inverse 
                           design is a computer system that utilizes inverse design and cloud 
                           computing to discover new and innovative materials. Inverse design 
                           involves starting with the desired properties or functionality of a 
                           material and then determining the appropriate composition and structure 
                           to achieve those properties. The cloud-based aspect of the system allows 
                           for the processing of large amounts of data and computation-intensive simulations, 
                           enabling the rapid discovery and optimization of novel materials. The use of machine 
                           learning algorithms can further enhance the system's ability to identify and 
                           explore new materials. This system has the potential to revolutionize the way 
                           new materials are discovered and developed, leading to new and improved materials 
                           for a wide range of applications.
                          </div>  
        
                         <br>
                            """, unsafe_allow_html=True)

        st.markdown(
            """
            <div>
            <h4 style='font-size: 25px; text-align:center; font-weight: 700;margin: 0 0 20px 0;color: #599bb3; background-color: purple'>References</h4>
            <p>
                <span style='font-size: 20px;font-weight:Bold; color: #012970;'>Hasan M Jamil, Lan Li -</span>
                <span style='font-size: 15px; font-weight:Bold; color: #012970; font-style: italic;'>A Knowledgebased Novel Materials Design System using Machine Learning</span>
            </p>
            <p>
                <span style='font-size: 20px;font-weight:Bold; color: #012970;'>Austin Biaggne, Lawrence Spear, German Barcenas, Maia Ketteridge, Young C.Kim, Joseph S.Melinger, Willia B.Knowlton, Bernard Yurke, and Lan Li - </span>
                <span style='font-size: 15px; font-weight:Bold; color: #012970; font-style: italic;'>Data-Driven and Multiscale Modeling of DNA-Templated Dye</span>   
            </p>
        
            </div>
            """, unsafe_allow_html=True
        )
        components.html(
            '''
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
        
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>
            <style>
                /*--------------------------------------------------------------
                # Footer
                --------------------------------------------------------------*/
                .footer .copyright {
                text-align: center;
                padding: 30px;
                color: #012970;
                }
        
                .footer .credits {
                padding: 10px;
                text-align: center;
                font-size: 13px;
                color: #012970;
                }
            </style>
            <footer id="footer" class="footer">
                <div class="container">
                <div class="copyright">
                    &copy; Copyright <strong><span>jamil@uidaho.edu</span></strong>. All Rights Reserved
                </div>
                </div>
            </footer>
        '''
        )
