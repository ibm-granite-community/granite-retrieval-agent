"""
requirements: crewai==0.102.0, crewai-tools==0.33.0
"""

import logging
from typing import Optional, Callable, Awaitable, List

from crewai import Crew, Process, Agent, Task, LLM
from crewai.tools import tool
from fastapi import Request
from open_webui.routers import retrieval
from open_webui.models.knowledge import KnowledgeTable
from open_webui import config as open_webui_config
from pydantic import BaseModel, Field


####################
# Prompts
####################

DEFAULT_ITEM_IDENTIFIER_GOAL = """The following is the description of an image.
Please enumerate in detail all the components of the image."""

DEFAULT_INSTRUCTION = f"{DEFAULT_ITEM_IDENTIFIER_GOAL} For every item or concept described in the image, find me at least 2 relevant\
    articles that describe the concept in detail, for educational purposes and teach me about them, citing references."

DEAULT_IMAGE_VERBALIZER_PROMPT = """Analyze the following image and provide a detailed description. Your response should cover the following aspects:
    1. Image Overview: Begin with a brief summary of the entire image. Describe the scene, the context, and the general mood or atmosphere it conveys.
    2. Objects and Entities: Identify and describe each significant object or entity present in the image. Include details such as: 
         - Name/Type: What is the object? If it's a person, animal, or inanimate object, mention its category.
         - Appearance: Describe its shape, size, and any distinctive features.
         - Color: Note the dominant and secondary colors.
         - Position: Where in the image is the object located? Is it central, off to the side, or in the background/foreground?
    3. Interrelations: Explain how the objects interact with each other or the environment. Are they in motion? Is there any evidence of interaction between them? 
    4. Patterns and Textures: Identify any repetitive patterns or textures present in the image. 
    5. Background and Environment: Describe the setting or backdrop of the image. Is it a natural landscape, an urban scene, an abstract space, or something else? 
    6. Symbols or Indicators: If there are any symbols, signs, text, or other indicators that could provide additional context, please mention them.
    7. Technical Elements (for diagrams or technical images): If applicable, describe the graphical elements, including lines, shapes, annotations, and any scaling indicators. 
Your goal is to create a thorough, nuanced description that another LLM could use as a starting point for further research or analysis about the content, context, and composition of the image.
Be sure to describe ALL prominent aspects of this image; do not miss any.
"""

ITEM_IDENTIFIER_PROMPT = """
    You are an item identifier.You will be given a description of an image, and your job is to identify all items and concepts that are part of the 
    image that will need to be researched in order to accomplish the goal.
    You will limit the number of items to the {item_limit} most important items pertaining to the image that will accopmlish the goal.
    You will not perform the research yourself, but will work with a helper who will perform the research. The helper has the following capabilities:
    1. Genearl generative AI capabilities.
    2. Search the internet
    3. Search the user's document store
    When giving out research tasks, please constrain the instructions to be within what the helper is capable of, and nothing beyond."""

ASSISTANT_PROMPT = """
    Make sure to provide a thorough answer that directly addresses the message you received.
    If the task is able to be accomplished without using tools, then do not make any tool calls.
    
    # Tool Use
    You have access to the following tools. Only use these available tools and do not attempt to use anything not listed - this will cause an error.
    When suggesting tool calls, please respond with a JSON for a function call with its proper arguments. Use non-escaped double quotes in the JSON.
    When you are using knowledge and web search tools to complete the instruction, answer the instruction only using the results from the search; do no supplement with your own knowledge.
    Never answer the instruction using links to URLs that were not discovered during the use of your search tools. Only respond with document links and URLs that your tools returned to you.
    Also make sure to provide the URL for the page you are using as your source or the document name.
    """


class Pipe:
    class Valves(BaseModel):
        TASK_MODEL_ID: str = Field(default="ollama/granite3.2:8b-instruct-q8_0")
        VISION_MODEL_ID: str = Field(
            default="ollama/granite3.2-vision:2b"
        )
        OPENAI_API_URL: str = Field(default="http://localhost:11434")
        OPENAI_API_KEY: str = Field(default="ollama")
        VISION_API_URL: str = Field(default="http://localhost:11434")
        MODEL_TEMPERATURE: float = Field(default=0)
        MAX_RESEARCH_CATEGORIES: int = Field(default=4)
        MAX_RESEARCH_ITERATIONS: int = Field(default=6)
        INCLUDE_KNOWLEDGE_SEARCH: bool = Field(default=False)
        RUN_PARALLEL_TASKS: bool = Field(default=False)


    def get_provider_models(self):
        return [
            {"id": self.valves.TASK_MODEL_ID, "name": self.valves.TASK_MODEL_ID},
        ]

    def __init__(self):
        self.type = "pipe"
        self.id = "granite_image_researcher"
        self.name = "Granite Image Researcher Agent"
        self.valves = self.Valves()

    def is_open_webui_request(self, body):
        """
        Checks if the request is an Open WebUI task, as opposed to a user task
        """
        message = str(body[-1])

        prompt_templates = {
            open_webui_config.DEFAULT_RAG_TEMPLATE.replace("\n", "\\n"),
            open_webui_config.DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
            open_webui_config.DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE.replace(
                "\n", "\\n"
            ),
        }

        for template in prompt_templates:
            if template is not None and template[:50] in message:
                return True

        return False

    async def emit_event_safe(self, message):
        event_data = {
            "type": "message",
            "data": {"content": message + "\n"},
        }
        try:
            await self.event_emitter(event_data)
        except Exception as e:
            logging.error(f"Error emitting event: {e}")

    async def pipe(
        self,
        body,
        __user__: Optional[dict],
        __request__: Request,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:

        # Grab env variables
        default_model = self.valves.TASK_MODEL_ID
        base_url = self.valves.OPENAI_API_URL
        api_key = self.valves.OPENAI_API_KEY
        vision_model = self.valves.VISION_MODEL_ID
        vision_url = self.valves.VISION_API_URL
        model_temp = self.valves.MODEL_TEMPERATURE
        max_research_categories = self.valves.MAX_RESEARCH_CATEGORIES
        max_research_iters = self.valves.MAX_RESEARCH_ITERATIONS
        include_knoweldge_search = self.valves.INCLUDE_KNOWLEDGE_SEARCH
        run_parallel_tasks = self.valves.RUN_PARALLEL_TASKS
        self.event_emitter = __event_emitter__
        self.owui_request = __request__
        self.user = __user__

        ##################
        # Crew LLM Config
        ##################
        class ResearchItem(BaseModel):
            item_name: str
            research_instructions: str

        class ResearchItems(BaseModel):
            items: List[ResearchItem]

        llm = LLM(
            model=default_model,
            base_url=base_url,
            api_key=api_key,
            temperature=model_temp,
        )

        vision_llm = LLM(
            model=vision_model,
            base_url=vision_url,
        )

        ##################
        # Check if this request is utility call from OpenWebUI
        ##################
        if self.is_open_webui_request(body["messages"]):
            print("Is open webui request")
            reply = llm.call([body["messages"][-1]])
            return reply

        ##################
        # Tool Definitions
        ##################

        @tool("WebSearch")
        def do_web_search(search_instructions: str) -> str:
            """Use this to search the internet. To use, provide a detailed search instruction that incorporates specific features, goals, and contextual details related to the query.
            Identify and include relevant aspects from any provided context, such as key topics, technologies, challenges, timelines, or use cases.
            Construct the instruction to enable a targeted search by specifying important attributes, keywords, and relationships within the context.
            """
            result = retrieval.search_web(
                self.owui_request,
                self.owui_request.app.state.config.RAG_WEB_SEARCH_ENGINE,
                search_instructions,
            )
            return str(result)

        @tool("Knowledge Search")
        def do_knowledge_search(search_instruction: str) -> str:
            """Use this tool if you need to obtain information that is unique to the user and cannot be found on the internet.
            Given an instruction on what knowledge you need to find, search the user's documents for information particular to them, their projects, and their domain.
            This is simple document search, it cannot perform any other complex tasks.
            This will not give you any results from the internet. Do not assume it can retrieve the latest news pertaining to any subject.
            """
            if not search_instruction:
                return "Please provide a search query."

            # First get all the user's knowledge bases associated with the model
            knowledge_item_list = KnowledgeTable().get_knowledge_bases()
            if len(knowledge_item_list) == 0:
                return "You don't have any knowledge bases."
            collection_list = []
            for item in knowledge_item_list:
                collection_list.append(item.id)

            collection_form = retrieval.QueryCollectionsForm(
                collection_names=collection_list, query=search_instruction
            )

            response = retrieval.query_collection_handler(
                request=self.owui_request, form_data=collection_form, user=self.user
            )
            messages = ""
            for entries in response["documents"]:
                for line in entries:
                    messages += line

            return messages

        #########################
        # Crew Agent Config
        #########################

        # Item Identifier Crew
        item_identifier = Agent(
            role="Item Identifier",
            goal="Identify which concepts that are in a described image need to be researched in order to accomplish the goal.",
            backstory=ITEM_IDENTIFIER_PROMPT,
            llm=llm,
            verbose=True,
        )
        identification_task = Task(
            description="Thorougly identify all items and concepts that are part of the image that will need to be researched in order to accomplish the goal. Goal: {goal} \n Image Description: {image_description}",
            agent=item_identifier,
            expected_output="A list of items and concepts that need to be researched in order to accomplish the goal.",
            output_pydantic=ResearchItems,
        )
        identifier_crew = Crew(
            agents=[item_identifier],
            tasks=[identification_task],
            process=Process.sequential,
            share_crew=False,
            verbose=True,
        )

        # Research Crew
        available_tools = [do_web_search]
        if include_knoweldge_search:
            available_tools.append(do_knowledge_search)
        researcher = Agent(
            role="Researcher",
            goal="You will be given a step/instruction to accomplish. Fully answer the instruction/question using document search or web search tools as necessary.",
            backstory=ASSISTANT_PROMPT,
            llm=llm,
            verbose=True,
            max_iter=max_research_iters,
            tools=available_tools,
        )
        resarch_task = Task(
            description="Fulfill the instruction given. {item_name} {research_instructions}",  # Keep in mind the previously gathered data from previous steps: {previously_executed_steps}',
            agent=researcher,
            expected_output="Information that directly answers the instruction given. If your answer references websites or documents, provide in-line citations in the form of hyperlinks for every reference.",
        )
        research_crew = Crew(
            agents=[researcher],
            tasks=[resarch_task],
            share_crew=False,
            verbose=True,
        )

        #########################
        # Begin Agentic Workflow
        #########################
        chat_history_text = ""
        image_info = []
        latest_instruction = ""

        def identify_message_content(message_content):
            """
            This function serves the purpose of parsing out text vs image content in the chat history.
            """
            content = {"text": "", "images": []}
            if type(message_content) == str:
                content["text"] = message_content

            else:
                for mc in message_content:
                    if mc["type"] == "image_url":
                        content["images"].append(mc)
                    elif mc["type"] == "text":
                        content["text"] += mc["text"]
                    else:
                        print(f"Ignoring content with type {mc['type']}")
            return content

        # Identify the latest instruction from the user's message history
        latest_content = identify_message_content(body["messages"][-1]["content"])
        if latest_content["text"]:
            latest_instruction = latest_content["text"]
        if latest_content["images"]:
            image_info = latest_content["images"]

        # Identify contextual chat history
        if len(body["messages"]) > 1:
            for i in range(0, len(body["messages"]) - 1):
                identified_content = identify_message_content(
                    body["messages"][i]["content"]
                )
                if identified_content["images"]:
                    image_info.extend(identified_content["images"])
                if identified_content["text"]:
                    chat_history_text += identified_content["text"]

        # For every image, whether in the latest instruction or the chat history, generate a description
        image_descriptions = []
        for i in range(len(image_info)):
            await self.emit_event_safe(message="Analyzing image...")
            image_query = DEAULT_IMAGE_VERBALIZER_PROMPT
            if latest_instruction:
                image_query += f"\n\nUse the following instruction from the user to further guide you: {latest_instruction}"
            if chat_history_text:
                image_query += f"\n\nAlso use the previous chat history to further guide you: {chat_history_text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_query},
                        image_info[i],
                    ],
                }
            ]
            image_description = vision_llm.call(messages)
            image_descriptions.append(
                f"Accompanying image description: {image_description}"
            )

        # Start identifying research goals according to the image description
        await self.emit_event_safe("Creating a resesarch plan...")
        identifier_goal = DEFAULT_ITEM_IDENTIFIER_GOAL
        if latest_instruction:
            identifier_goal += f"\n\nUse the following instruction from the user to further guide you: {latest_instruction}"
        if chat_history_text:
            identifier_goal += f"\n\nAlso use the previous chat history to further guide you: {chat_history_text}"
        inputs = {"goal": identifier_goal, "image_description": image_descriptions, "item_limit": max_research_categories}
        identifier_crew.kickoff(inputs)

        # Now that we've established our research targets, start a parallel async crew of researchers to tackle it
        await self.emit_event_safe("Researching items...")
        tasks = []
        for task in identification_task.output.pydantic.items:
            tasks.append(
                {
                    "item_name": task.item_name,
                    "research_instructions": task.research_instructions,
                }
            )
        
        if run_parallel_tasks:
            outputs = await research_crew.kickoff_for_each_async(tasks)
        else:
            outputs = research_crew.kickoff_for_each(tasks)

        # Create the final report
        await self.emit_event_safe("Summing up findings...")
        final_output = llm.call(
            f"{{Thoroughly answer the user's question, providing links to all URLs and documents used in your response. You may only use the following information to answer the question. If no reference URLs exist, do not fabricate them. If the following information does not have all the information you need to answer all aspects of the user question, then you may highlight those aspects. User query: {DEFAULT_INSTRUCTION} \n\n Image description:  {image_descriptions} \n\n Gathered information: {outputs}}}"
        )
        await self.emit_event_safe("(If results don't show soon, refresh)")
        return final_output
