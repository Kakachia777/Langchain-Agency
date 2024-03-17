import os
import asyncio
import base64
from typing import Optional
from urllib.parse import urlparse
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import DeepInfra
from crewai import Crew, Agent, Process, Task
from crewai_tools import tool
from langchain_community.tools.playwright import (
    ClickTool,
    NavigateTool,
    NavigateBackTool,
    ExtractTextTool,
    ExtractHyperlinksTool,
    GetElementsTool,
    CurrentWebPageTool,
)
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image, ImageDraw
from io import BytesIO

import nest_asyncio
nest_asyncio.apply()

# Create asynchronous Playwright browser (headless=False for visualization)
sync_browser = create_sync_playwright_browser(headless=False)

@tool("Screenshot Tool")
def screenshot_tool(description: str) -> str:
    """Captures a screenshot of the current webpage and returns it as a base64-encoded string."""
    screenshot_path = "screenshot.png"
    page = sync_browser.new_page()
    page.screenshot(path=screenshot_path)
    with open(screenshot_path, "rb") as f:
        image_data = f.read()
    os.remove(screenshot_path)
    return base64.b64encode(image_data).decode()

class AsyncNavigateTool:
    name = "navigate_browser"
    description = "Navigate a browser to the specified URL"
    sync_browser = sync_browser

    async def _arun(self, query: dict) -> str:
        url = query.get("url")
        page = await self.sync_browser.new_page()
        await page.goto(url)
        return f"Navigating to {url} returned status code {page.status_code}"

load_dotenv()

# Set up Google API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up DeepInfra API token
DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN")

def create_custom_llm():
    while True:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                verbose=True,
                temperature=0.3,
                google_api_key=GOOGLE_API_KEY,
            )
            return llm
        except Exception as e:
            print(f"Error occurred while creating the language model: {str(e)}. Retrying...")
            continue

# Initialize the Gemini model
gemini_llm = create_custom_llm()

# Initialize the LLaVA-1.5 model
llava_llm = DeepInfra(model_id="llava-hf/llava-1.5-7b-hf")
llava_llm.model_kwargs = {
    "temperature": 0.3,
    "repetition_penalty": 1.2,
    "max_new_tokens": 450,
    "top_p": 0.9,
}

# Define the tools
tools = [
    ClickTool(sync_browser=sync_browser),
    NavigateTool(sync_browser=sync_browser),
    NavigateBackTool(sync_browser=sync_browser),
    ExtractTextTool(sync_browser=sync_browser),
    ExtractHyperlinksTool(sync_browser=sync_browser),
    GetElementsTool(sync_browser=sync_browser),
    NavigateBackTool(sync_browser=sync_browser),
    CurrentWebPageTool(sync_browser=sync_browser),
    screenshot_tool,  # Add the custom screenshot_tool
]

# Create CrewAI agents
researcher = Agent(
    name="Researcher",
    tools=tools,
    llm=gemini_llm,
    role="Researcher",
    goal="Search for relevant links based on the user's goal",
    backstory="A skilled researcher with expertise in web search and information retrieval, but first always uses google platform.",
    verbose=True,
    allow_delegation=True,
)

navigator = Agent(
    name="Navigator",
    tools=tools,
    llm=gemini_llm,
    role="Navigator",
    goal="Navigate to one of the links provided by the Researcher, choose one of the links and navigate to it",
    backstory="An experienced web navigator capable of efficiently browsing and exploring webpages.",
    verbose=True,
    allow_delegation=True,
)

visual_analyzer = Agent(
    name="Visual Analyzer",
    tools=tools,
    llm=llava_llm,
    role="Visual Analyzer",
    goal="Analyze the visual elements on the webpage and provide guidance on which elements to interact with",
    backstory="A proficient visual analyzer with the ability to understand and interpret visual information on webpages.",
    verbose=True,
    allow_delegation=True,
)

extractor = Agent(
    name="Extractor",
    tools=tools,
    llm=llava_llm,
    role="Extractor",
    goal="Extract relevant content from the webpage, provided from Navigator",
    backstory="A proficient content extractor with the ability to identify and extract relevant information from webpages.",
    verbose=True,
    allow_delegation=True,
)

summarizer = Agent(
    name="Summarizer",
    tools=tools,
    llm=llava_llm,
    role="Summarizer",
    goal="Summarize the collected information and determine if the goal is reached, if not, you can navigate back",
    backstory="A skilled summarizer with the ability to condense information and assess goal achievement.",
    verbose=True,
    allow_delegation=True,
)

# Define tasks for each agent
search_task = Task(
    description="Search for relevant links based on the user's goal: {goal}",
    expected_output="A list of 5 links, then choosing one of the link for Navigator",
    agent=researcher,
)

navigate_task = Task(
    description="Navigate to the link provided by the Researcher for the goal: {goal}",
    expected_output="Successful navigation to one of the link",
    agent=navigator,
)

visual_analysis_task = Task(
    description="Analyze the visual elements on the webpage and provide guidance on which elements to interact with",
    expected_output="A list of numbered bounding boxes for interactive elements on the webpage",
    agent=visual_analyzer,
)

extract_task = Task(
    description="Extract relevant content from the webpages related to the goal: {goal}",
    expected_output="Extracted content from the webpage",
    agent=extractor,
)

summarize_task = Task(
    description="Summarize the collected information and determine if the goal '{goal}' is reached, if not you can navigate back",
    expected_output="Summary of the collected information and goal status",
    agent=summarizer,
)

crew = Crew(
    agents=[researcher, navigator, visual_analyzer, extractor, summarizer],
    tasks=[search_task, navigate_task, visual_analysis_task, extract_task, summarize_task],
    process="hierarchical",
    verbose=True,
    manager_llm=gemini_llm,
)

# Define the crawling task
def extract_links(search_results):
    links = []
    for result in search_results:
        url = result.get("link")
        if url and urlparse(url).scheme in ["http", "https"]:
            links.append(url)
    return links

def _parse_response_candidate(candidate, stream=False):
    try:
        first_part = response_candidate.content.parts[0]
        return first_part.content
    except (IndexError, AttributeError):
        # Handle the case when the response format is unexpected
        return ""

def calculate_similarity(text1, text2):
    # Convert the texts to feature vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    
    # Calculate the cosine similarity between the vectors
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    
    return similarity

@tool("Bounding Box Tool")
def bounding_box_tool(description: str) -> str:
    """Adds bounding boxes and numbering to elements on the current webpage and returns a base64-encoded image."""
    page = sync_browser.new_page()
    selector = "button, a, input, [role='button'], [role='link']"
    elements = page.query_selector_all(selector)
    
    viewport_size = page.evaluate('() => ({width: document.documentElement.clientWidth, height: document.documentElement.clientHeight})')
    screenshot_path = "screenshot_with_boxes.png"
    page.screenshot(path=screenshot_path, full_page=True)
    
    with Image.open(screenshot_path) as screenshot:
        draw = ImageDraw.Draw(screenshot)
        for i, element in enumerate(elements):
            box = element.bounding_box()
            if box:
                draw.rectangle((box["x"], box["y"], box["x"]+box["width"], box["y"]+box["height"]), outline="red", width=2)
                draw.text((box["x"], box["y"]-20), str(i+1), fill="red")
        
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        screenshot_str = base64.b64encode(buffered.getvalue()).decode()
        
    os.remove(screenshot_path)
    return f"data:image/png;base64,{screenshot_str}"

tools.append(bounding_box_tool)

async def crawl_task(goal, link=None):
    collected_info = ""
    visited_urls = set()
    
    if link is None:
        # Search for relevant links on Google
        search_query = f"Google search for {goal}"
        search_result = crew.kickoff(inputs={"goal": search_query})
        links = extract_links(search_result)
    else:
        links = [link]
    
    async with sync_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        for link in links:
            if link in visited_urls:
                continue
            
            visited_urls.add(link)
            
            try:
                # Navigate to the link
                navigation_result = crew.kickoff(inputs={"goal": goal, "link": link})
                await page.goto(link)
                
                # Capture a screenshot of the webpage
                screenshot_base64 = await crew.kickoff(tools=[screenshot_tool])
                
                # Add bounding boxes and numbering to the screenshot
                bounding_boxes_base64 = await crew.kickoff(tools=[bounding_box_tool])
                
                # Analyze the visual elements on the webpage
                visual_analysis_result = crew.kickoff(inputs={"screenshot": screenshot_base64, "bounding_boxes": bounding_boxes_base64})
                
                # Extract the bounding boxes and element numbers from the visual analysis result
                bounding_boxes = extract_bounding_boxes(visual_analysis_result)
                
                # Interact with the elements based on the bounding boxes
                for element_number in bounding_boxes:
                    element = await page.locator(f"[data-element-number='{element_number}']")
                    await element.click()
                    
                    # Extract the content from the webpage after clicking the element
                    extraction_result = crew.kickoff(inputs={"goal": goal})
                    
                    # Calculate the similarity between the extracted content and the goal
                    similarity = calculate_similarity(extraction_result, goal)
                    
                    # Check if the similarity is above a certain threshold (e.g., 0.5)
                    if similarity > 0.5:
                        # Summarize the collected information and determine if the goal is reached
                        summary_result = crew.kickoff(inputs={"goal": goal, "collected_info": collected_info})
                        
                        if "goal reached" in summary_result.lower():
                            await browser.close()
                            return summary_result
                        else:
                            # Navigate back to the previous page
                            await page.go_back()
                            break
                    
                    # Navigate back to the previous page
                    await page.go_back()
                
                # Extract hyperlinks from the current page
                hyperlinks = crew.kickoff(inputs={"goal": "Extract hyperlinks from the current page"})
                
                # Recursively crawl the hyperlinks
                for hyperlink in hyperlinks.split("\n"):
                    result = await crawl_task(goal, hyperlink)
                    if result:
                        await browser.close()
                        return result
                
            except Exception as e:
                print(f"Error occurred while processing {link}: {str(e)}")
                continue
        
        await browser.close()
    
    # If no satisfactory information is found, provide a summary of the collected information
    summary_result = crew.kickoff(inputs={"goal": goal, "collected_info": collected_info})
    return summary_result

# Create an asynchronous entry point
async def main():
    goal = input("Please enter your search goal: ")
    result = await crawl_task(goal)
    print(result)

# Run the script
asyncio.get_event_loop().run_until_complete(main())