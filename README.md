# genai-task

To run this code please ensure that all of the python modules have been downloaded and that the directory has been set up with genai-task.py in a folder with a separate folder inside that main folder called docs which contains the ml_project1_data.csv file.

This code can then be run on the command line using the following command
Python3 genai-task.py <query>

The data_ingestor.py was a file that I was going to use to create a vectordb that would then be saved to the local disk which did not pan out. 
Instead I moved this over to the main python file. 

Unfortunately, my code is not fully functioning, this is because I was unable to correctly link the Azure OpenAI model to the datafile. I was unable to create this link as due to the fact that I am on a limited burner account for Azure I was unable to create the required cognitive search resource on Azure that would allow the link to be made to the dataset that the queries were to be made on. 

If I had access to an OpenAI endpoint then I am fairly sure I would have been able to get the code working, I would have added in the correct connection and embedding and the code on line 72 would have been able to correctly connect the vectordb to the OpenAI generative AI. 
