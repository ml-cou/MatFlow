import streamlit as st
from subpage.navbar import navbar
st.set_page_config(layout='wide',page_title='Contact', initial_sidebar_state='collapsed')

def form():
    navbar()
    st.markdown(
        '''
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/flowbite/1.6.3/flowbite.min.css" rel="stylesheet" />
    
            <script src="https://cdnjs.cloudflare.com/ajax/libs/flowbite/1.6.3/flowbite.min.js"></script>
    
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>
            <style>
              .contact .info-box {
              color: #444444;
              background: #fafbff;
              padding: 30px;
            }
    
            .contact .info-box i {
              font-size: 38px;
              line-height: 0;
              color: #4154f1;
            }
    
            .contact .info-box h3 {
              font-size: 20px;
              color: #012970;
              font-weight: 700;
              margin: 20px 0 10px 0;
            }
    
            .contact .info-box p {
              padding: 0;
              line-height: 24px;
              font-size: 14px;
              margin-bottom: 0;
            }
            </style>
          <section id="contact" class="contact">
            <div class="container" data-aos="fade-up">
              <header class="section-header">
                <h2 style= 'text-align:center; font-size: 45px; font-weight: bold; color:#0e9099'>
                Contact Us</h2>
              </header>
              <div class="row gy-12">
                <div class="col-lg-12">
                  <div class="row gy-12">
                  <div class="col-md-6">
                      <div class="info-box">
                        <p><h3 style='text-align: center; background-color:#fbffe3; font-size: 25px; 
                                   font-family: "Lucida Console", "Courier New", monospace; 
                                   font-weight: bold; color:#0e9099;'>Hasan Jamil, Ph.D.</p><br>           
                        <p style='display: block; ont-family: "Lucida Console", "Courier New", monospace; 
                        font-size: 20px; text-align: center; font-weight: bold; color:purple'>
                        Department of Computer Science
                        <br>Associate Professor
                        <br>University of Idaho </p>
                      </div>
                    </div>
                    <div class="col-md-6">
                      <div class="info-box">
                        <i class="bi bi-geo-alt"></i>
                        <p><h3 style='text-align: center; font-size: 25px; 
                                   font-family: "Lucida Console", "Courier New", monospace; 
                                   font-weight: bold; color:#0e9099;'>Md. Hasan Hafizur Rahman</p><br>           
                        <p style='display: block; ont-family: "Lucida Console", "Courier New", monospace; 
                        font-size: 20px; text-align: center; font-weight: bold; color:purple'>
                        Department of Computer Science and Engineering
                        <br>Comilla University
                        <br> Cumilla - 3506, Bangladesh
                        </p>
                      </div>
                    </div>
                    <div class="col-md-6">
                      <div style="padding: 0 00px 0 00px">
                          <form id="Form style="padding: 0 00px 0 00px" class="border-2 border-gray-300 appearance-none dark:text-white dark:border-gray-600 dark:focus:border-blue-500 focus:outline-none focus:ring-0 focus:border-blue-600 peer p-1 rounded" style="border-color: rgb(28 100 242)">
                              <div class="mb-1">
                                  <h1 class="text-center text-blue-600 text-3xl">Contact us</h1>
                              </div>
                              <div class="relative" >
                                  <input type="text" id="floating_outlined" name="firstName"
                                      onchange="onChangeHandling(this)"
                                      class="block px-2.5 pb-2 pt-3 w-full text-sm text-gray-900 bg-transparent rounded-lg border-2 border-gray-300 appearance-none dark:text-white dark:border-gray-600 dark:focus:border-blue-500 focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder="" />
                                  <label for="floating_outlined" style
                                      class="absolute text-sm text-gray-500 dark:text-gray-400 duration-300 transform -translate-y-3 scale-75 top-2 z-10 origin-[0] bg-white dark:bg-gray-900 px-2 peer-focus:px-2 peer-focus:text-blue-600 peer-focus:dark:text-blue-500 peer-placeholder-shown:scale-100 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:top-1/2 peer-focus:top-2 peer-focus:scale-75 peer-focus:-translate-y-3">First Name</label>
                              </div>
                              <br />
                              <div class="relative" >
                                  <input type="text" id="floating_outlined" name="lastName"
                                      onchange="onChangeHandling(this)"
                                      class="block px-2.5 pb-2 pt-3 w-full text-sm text-gray-900 bg-transparent rounded-lg border-2 border-gray-300 appearance-none dark:text-white dark:border-gray-600 dark:focus:border-blue-500 focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder="" />
                                  <label for="floating_outlined" style
                                      class="absolute text-sm text-gray-500 dark:text-gray-400 duration-300 transform -translate-y-3 scale-75 top-2 z-10 origin-[0] bg-white dark:bg-gray-900 px-2 peer-focus:px-2 peer-focus:text-blue-600 peer-focus:dark:text-blue-500 peer-placeholder-shown:scale-100 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:top-1/2 peer-focus:top-2 peer-focus:scale-75 peer-focus:-translate-y-3">Last Name</label>
                              </div>
                              <br />
                              <div class="relative" >
                                  <input type="text" id="floating_outlined" name="email"
                                      onchange="onChangeHandling(this)"
                                      class="block px-2.5 pb-2 pt-3 w-full text-sm text-gray-900 bg-transparent rounded-lg border-2 border-gray-300 appearance-none dark:text-white dark:border-gray-600 dark:focus:border-blue-500 focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder="" />
                                  <label for="floating_outlined" style
                                      class="absolute text-sm text-gray-500 dark:text-gray-400 duration-300 transform -translate-y-3 scale-75 top-2 z-10 origin-[0] bg-white dark:bg-gray-900 px-2 peer-focus:px-2 peer-focus:text-blue-600 peer-focus:dark:text-blue-500 peer-placeholder-shown:scale-100 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:top-1/2 peer-focus:top-2 peer-focus:scale-75 peer-focus:-translate-y-3">Email</label>
                              </div>
                              <br />
                              <div class="relative" >
                                  <textarea id="floating_outlined" name="desc" type="textarea" 
                                      onchange="onChangeHandling(this)"
                                      class="block px-2.5 pb-2 pt-3 w-full text-sm text-gray-900 bg-transparent rounded-lg border-2 border-gray-300 appearance-none dark:text-white dark:border-gray-600 dark:focus:border-blue-500 focus:outline-none focus:ring-0 focus:border-blue-600 peer" placeholder=""
                                      >
                                  </textarea>
                                  <label for="floating_outlined" style
                                      class="absolute text-sm text-gray-500 dark:text-gray-400 duration-300 transform -translate-y-3 scale-75 top-2 z-10 origin-[0] bg-white dark:bg-gray-900 px-2 peer-focus:px-2 peer-focus:text-blue-600 peer-focus:dark:text-blue-500 peer-placeholder-shown:scale-100 peer-placeholder-shown:-translate-y-1/2 peer-placeholder-shown:top-1/2 peer-focus:top-2 peer-focus:scale-75 peer-focus:-translate-y-3">Description</label>
                              </div>
                              <br />
                              <div
                                  onclick="onSubmit()"
                                  class="relative inline-flex items-center justify-center p-0.5 mb-2 mr-2 overflow-hidden text-sm font-medium text-gray-900 rounded-lg group bg-gradient-to-br from-green-400 to-blue-600 group-hover:from-green-400 group-hover:to-blue-600 hover:text-white dark:text-white focus:ring-4 focus:outline-none focus:ring-green-200 dark:focus:ring-green-800" style="margin-left: 200px">
                              <span 
                                  class="relative px-5 py-2.5 transition-all ease-in duration-75 bg-white dark:bg-gray-900 rounded-md group-hover:bg-opacity-0">
                                  Submit
                              </span>
                              </div>
                          </form>
                      </div>
                    </div>
                    <div class="col-md-6">
                      <div class="info-box">
                        <i class="bi bi-envelope"></i>
                        <h3 style='text-align: center; font-size: 25px; 
                                   font-family: "Lucida Console", "Courier New", monospace; 
                                   font-weight: bold; color:#0e9099;'>📧 email Us
                                   </h3>
                        <p style='text-align: center; font-size: 15px; font-style: italic;
                                   font-family: "Lucida Console", "Courier New", monospace; 
                                   font-weight: bold; color:#0e9099;'>hhr@cou.ac.bd<br>hasancse03@gmail.com</p>
                      </div>
                      <div class="info-box">
                        <i class="bi bi-telephone"></i>
                        <h3 style='text-align: center; font-size: 25px; 
                                   font-family: "Lucida Console", "Courier New", monospace; 
                                   font-weight: bold; color:#0e9099;'>☎ Voice Call
                                   <br>♦+880 1721 0 2 1 9 0 9♦</h3>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <script>
                const formData = {
                    firstName : "",
                    lastName : "",
                    email : "",
                    desc : ""
                }
    
                function onChangeHandling(e) {        
                    formData[e.name] = e.value
                    console.log("clicked, ", e.name, formData)
                }
                function onSubmit() {
                    console.log(formData, "d")
                }
            </script>
        ''', unsafe_allow_html=True
    )

form()
