# Agency using CrewAI, DeepInfra and Gemini Pro

![Alt Text](https://site-blog-files.s3.amazonaws.com/public/how-to-automate-processes-with-crewai/how-to-automate-processes-with-crewai.jpg)

This is a several Agency Bots where you need to provide instructions and it will handle all of the coding. This is accomplished with the assistance of CrewAI, DeepInfra and the Google Gemini-Pro model.

This provides the potential to create complex workflows with a variety of conditions. Something needs minor adjustments to function properly.

## Installation

* Clone the repo
* Install the required packages
```shell
pip install -r requirements.txt
```
* Create a .env file to add GOOGLE_API_KEY and DEEPINFRA_API_TOKEN 
```shell
GOOGLE_API_KEY = 'YOUR_API_KEY'
DEEPINFRA_API_TOKEN = 'YOUR_API_KEY'
```
* Run the Playwright Agents.py in the terminal
```shell
python run Playwright Agents.py
python run Marketing Agency.py
```

**_NOTE:_**   A sample input prompt is given in the folder, and some changes may be needed in the *output Python file*.


## Authors

- [@Kakachia777](https://github.com/Kakachia777)

