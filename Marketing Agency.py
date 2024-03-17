import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool

# Load environment variables
load_dotenv()

# Get Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize ChatGoogleGenerativeAI with Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
)

# Define search tool (flexible for each agent)
def create_search_tool(query):
    wrapper = DuckDuckGoSearchAPIWrapper(region="en-en", time="d", max_results=5)
    search = DuckDuckGoSearchRun()
    return Tool(
        name="research",
        func=search.run,
        description="Conduct research on the provided query.",
    )

# Define Agents (each with research capability)
entrepreneur = Agent(
    role="Entrepreneur",
    goal="Develop a viable business model",
    backstory="You are a visionary entrepreneur with a passion for innovation and a drive to create successful businesses.",
    verbose=True,
    tools=[create_search_tool("relevant query")],  # Dynamically create search tool
    allow_delegation=False,
    llm=llm,
)
market_researcher = Agent(
    role="Market Researcher",
    goal="Analyze market trends and identify opportunities",
    backstory="You are a skilled market researcher with expertise in gathering and interpreting data to identify market trends and potential opportunities.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)
product_developer = Agent(
    role="Product Developer",
    goal="Design and develop a desirable product or service",
    backstory="You are a talented product developer with a strong understanding of user needs and the ability to create innovative solutions.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)
marketing_specialist = Agent(
    role="Marketing Specialist",
    goal="Develop effective marketing strategies",
    backstory="You are a creative marketing specialist with expertise in crafting and executing successful marketing campaigns.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)
financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyze financial viability and create projections",
    backstory="You are a meticulous financial analyst with a keen understanding of financial models and forecasting.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
)

# Define Tasks
identify_opportunity_task = Task(
    description="Identify a promising market opportunity with potential for growth.",
    agent=market_researcher,
    expected_output="A detailed report outlining potential market opportunities.",
)

develop_product_task = Task(
    description="Design and develop a product or service that addresses the identified opportunity.",
    agent=product_developer,
    expected_output="A detailed product or service design document.",
)

marketing_strategy_task = Task(
    description="Develop a comprehensive marketing strategy to reach target customers.",
    agent=marketing_specialist,
    expected_output="A comprehensive marketing plan with specific strategies and tactics.",
)

financial_analysis_task = Task(
    description="Analyze the financial viability of the business model and create projections.",
    agent=financial_analyst,
    expected_output="A financial analysis report with projections and key financial metrics.",
)

refine_model_task = Task(
    description="Refine the business model based on insights from all agents.",
    agent=entrepreneur,
    expected_output="A revised business model document incorporating insights from all agents.",
)

# Create Crew
business_crew = Crew(
    agents=[
        entrepreneur,
        market_researcher,
        product_developer,
        marketing_specialist,
        financial_analyst,
    ],
    tasks=[
        identify_opportunity_task,
        develop_product_task,
        marketing_strategy_task,
        financial_analysis_task,
        refine_model_task,
    ],
    process=Process.sequential,
)

# Kick off the crew
result = business_crew.kickoff()
print(result)