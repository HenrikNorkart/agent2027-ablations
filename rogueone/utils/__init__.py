from rogueone.utils.config import (
    LLMConfig,
    rogueoneConfig,
    EmbeddingConfig,
    ExtractorAgentConfig,
    MedicalKnowledgeAgentConfig,
    TestAgentConfig,
    ScientistAgentConfig,
    ConsoleConfig,
    WandBConfig,
)


llm_cfg = LLMConfig()
rouge_one_cfg = rogueoneConfig()
embedding_cfg = EmbeddingConfig()
extractor_agent_cfg = ExtractorAgentConfig()
knowledge_agent_cfg = MedicalKnowledgeAgentConfig()
test_agent_cfg = TestAgentConfig()
scientist_agent_cfg = ScientistAgentConfig()
console_cfg = ConsoleConfig()
wandb_cfg = WandBConfig()

__all__ = [
    "llm_cfg",
    "rouge_one_cfg",
    "embedding_cfg",
    "extractor_agent_cfg",
    "knowledge_agent_cfg",
    "test_agent_cfg",
    "scientist_agent_cfg",
    "console_cfg",
    "wandb_cfg",
]
