import logging
import uuid
from typing import Optional

import numpy as np
from langchain_core.output_parsers import JsonOutputParser

from codeconcise.config.pipeline_config_service import PipelineConfigService
from codeconcise.llm.models.model_config import LlmModelConfig, EmbeddingsModelConfig
from codeconcise.llm.services.token_tracking_cc_lite_llm_wrapper import (
    TokenTrackingCCLiteLLMWrapper,
)
from codeconcise.metrics.observer.stage_observer import StageObserver
from codeconcise.pipeline.code_parser.comprehension_prompts import (
    CAPABILITY_MAPPING,
)
from codeconcise.pipeline.code_parser.comprehension_prompts import (
    ComprehensionPromptFactory,
)
from codeconcise.sdk.graph.graph_constructor import GraphClientBuilder
from codeconcise.sdk.models.action import Action
from codeconcise.sdk.models.capability import Capability
from codeconcise.sdk.models.code_summary import CodeSummary
from codeconcise.sdk.models.enriched_node import EnrichedNode
from codeconcise.sdk.models.nodes.base_cc_node import BaseCodeConciseNode
from codeconcise.sdk.tools.comprehension.dynamic_code_summary import CODE_SUMMARY
from codeconcise.sdk.utils.document import node_to_document

BUSINESS_CAPABILITY_MAPPING = "BusinessCapabilityMapping"
CAPABILITY_NODE_TYPE = "CAPABILITY"
CAPABILITY_SUMMARY_NODE_TYPE = "CAPABILITY_SUMMARY"
OUTPUT_PATH = "OUTPUT_PATH"
SOURCE_FILE = "source.pkl"
MAPPING_REQUESTS_FILE = "mapping_requests.jsonl"
MAPPING_RESPONSES_FILE = "mapping_responses.jsonl"
MAPPING_RESPONSES_ERRORS_FILE = "mapping_responses_errors.jsonl"
SUMMARY_REQUESTS_FILE = "capability_summary_requests.jsonl"
SUMMARY_RESPONSES_FILE = "capability_summary_responses.jsonl"
SUMMARY_RESPONSES_ERRORS_FILE = "capability_summary_responses_errors.jsonl"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
SKIP_CAPABILITY_WARNING = "Skipping business capability content: '''{content}''', as it's not fit in the provided business capabilities"
CAPABILITIES_REQUIRED_ERROR = "capabilities is required in additional_args"

logger = logging.getLogger(__name__)


class BusinessCapabilityMapping(Action):

    def __init__(
        self,
        name: str,
        package: str,
        db_name: str,
        additional_args: dict,
        observer: StageObserver,
    ):
        super().__init__(
            name=name,
            package=package,
            additional_args=additional_args,
            db_name=db_name,
            observer=observer,
        )
        if "capabilities" not in additional_args:
            raise ValueError(CAPABILITIES_REQUIRED_ERROR)
        self.config = PipelineConfigService.get_instance()

    async def run(
        self,
        nodes: list[BaseCodeConciseNode],
        llm_model: LlmModelConfig,
        embeddings_model: EmbeddingsModelConfig,
        language: str,
        graph_constructor: GraphClientBuilder,
    ) -> Optional[list[EnrichedNode]]:
        cc_litellm_wrapper = TokenTrackingCCLiteLLMWrapper(self.observer, self.name)
        documents = [node_to_document(node) for node in nodes]

        capabilities = self.additional_args.get("capabilities", [])
        valid_capabilities = [cap.lower() for cap in capabilities]
        capabilities_descriptions = self.additional_args.get(
            "capabilities_descriptions", []
        )

        prompts = []
        for document in documents:
            prompt = ComprehensionPromptFactory.get(
                prompt_type=CAPABILITY_MAPPING,
                params={
                    "language": language,
                    "additional_args": {
                        "capabilities": capabilities,
                        "capabilities_descriptions": capabilities_descriptions,
                    },
                    "file_content": document.page_content,
                },
            )
            system = prompt.SYSTEM_PROMPT
            prompt = prompt.QUESTION_GENERATOR_SYSTEM_PROMPT
            prompts.append(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )

        capability_responses = await cc_litellm_wrapper.batch_completion(prompts)
        enriched_nodes = []
        capability_texts = []
        caps = []

        for capability_response, document in zip(capability_responses, documents):
            capability_response.choices[0]["message"]["content"] = (
                capability_response.choices[0]["message"]["content"].strip()
            )
            content: str = capability_response.choices[0]["message"]["content"]

            content: dict = JsonOutputParser().parse(content)

            for c in content["matched_capabilities"]:
                if c["capability"].lower() in [*valid_capabilities]:
                    capability_texts.append(c["description"])
                    caps.append(
                        (document.metadata["id"], c["capability"], c["description"])
                    )

                else:
                    logger.warning(SKIP_CAPABILITY_WARNING.format(content=c))

        capability_texts_batch_size = int(np.ceil(len(capability_texts) / 100))
        capability_texts_batches = np.array_split(
            capability_texts, capability_texts_batch_size
        )
        capability_embeddings = []

        for capability_batch in capability_texts_batches:
            capability_embeddings += await cc_litellm_wrapper.generate_embeddings(
                list(map(str, capability_batch))
            )

        enriched_nodes += [
            Capability(
                id=cap,
                summary="",
                summaryVector=[],
                type=CAPABILITY_NODE_TYPE,
                related_to_id=[id],
                display_name=cap,
                generator=self.additional_args.get("generator", self.name),
                latestUpdateGenerator=self.additional_args.get("generator", self.name),
            )
            for (id, cap, summary), embeddings in zip(caps, capability_embeddings)
        ]
        enriched_nodes += [
            CodeSummary(
                id=str(uuid.uuid4()),
                summary=summary,
                summaryVector=embeddings,
                type=CODE_SUMMARY,
                related_to_id=[id],
                source_content="",
                codeVector=[],
                display_name=cap,
                generator=self.additional_args.get("generator", self.name),
                latestUpdateGenerator=self.additional_args.get("generator", self.name),
            )
            for (id, cap, summary), embeddings in zip(caps, capability_embeddings)
        ]

        return enriched_nodes

    async def post_processing(
        self,
        processed_nodes: list[BaseCodeConciseNode],
        db_name: str,
        graph_constructor: GraphClientBuilder,
        vector_dimensions: int,
    ) -> list[BaseCodeConciseNode]:

        graph_constructor.client().execute_query(
            lambda driver: driver.execute_query(
                """
                CREATE VECTOR INDEX `capability-summary-embeddings` IF NOT EXISTS
                FOR (n:CAPABILITY) ON (n.summaryVector)
                OPTIONS {indexConfig: {
                 `vector.dimensions`: $vector_dimensions,
                 `vector.similarity_function`: 'cosine'
                }}
                """,
                database_=db_name,
                vector_dimensions=vector_dimensions,
            )
        )

        graph_constructor.client().execute_query(
            lambda driver: driver.execute_query(
                """
                CREATE FULLTEXT INDEX `capability-summary-text` IF NOT EXISTS FOR (n:CAPABILITY) ON EACH [n.summary]
                """,
                database_=db_name,
            )
        )

        return processed_nodes
