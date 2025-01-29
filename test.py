# Warning control
import warnings
import os
from dotenv import load_dotenv


warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool
                         
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

# Connecting to an OpenAI-compatible LLM
llm = LLM(
    model="gpt-4o-mini",
    api_key=openai_api_key
)
###################################################################
docs_scrape_tool = ScrapeWebsiteTool(website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/")
###################################################################
support_agent = Agent(
    role="Senior Support Representative",
	goal="Be the most friendly and helpful support representative in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and are now working on providing support to {customer}, a super important customer for your company."
		"You need to make sure that you provide the best support! Make sure to provide full complete answers, and make no assumptions."
        
	),
    llm=llm,
	allow_delegation=False,
	verbose=True
)
###################################################################

support_quality_assurance_agent = Agent(
	role="Support Quality Assurance Specialist",
	goal="Get recognition for providing the best support quality assurance in your team",
	backstory=(
		"You work at crewAI (https://crewai.com) and are now working with your team on a request from {customer} ensuring that the support representative is providing the best support possible."
		"You need to make sure that the support representative is providing full complete answers, and make no assumptions."
	),
    llm=llm,
    allow_delegation=True,
	verbose=True
)
###################################################################

inquiry_resolution_task = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know to provide the best support possible."
		"You must strive to provide a complete and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the customer's inquiry that addresses all aspects of their question."
        "The response should include references to everything you used to find the answer, including external data or solutions. Ensure the answer is complete, leaving no questions unanswered, and maintain a helpful and friendly tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)


###################################################################

quality_assurance_review_task = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry have been addressed thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to find the information, ensuring the response is well-supported and leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response ready to be sent to the customer.\n"
        "This response should fully address the customer's inquiry, incorporating all relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution_task, quality_assurance_review_task],
  verbose=True,
  memory=True
)


inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "I need help with setting up a Crew and kicking it off, specifically how can I add memory to my crew? Can you provide guidance?"
}

result = crew.kickoff(inputs)
print(result)