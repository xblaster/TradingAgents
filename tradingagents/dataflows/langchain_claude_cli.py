"""
LangChain-compatible wrapper pour Claude CLI.

Ce module fournit une classe compatible avec l'interface LangChain ChatModel
pour utiliser Claude Code CLI comme backend LLM.
"""

from typing import Any, List, Optional, Iterator, Dict
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field

from .claude_cli import ClaudeCliService, ClaudeCliError


class ChatClaudeCLI(BaseChatModel):
    """
    Wrapper LangChain pour Claude CLI.

    Cette classe permet d'utiliser Claude Code CLI comme un modèle de chat
    compatible avec LangChain, en remplacement de ChatOpenAI.
    """

    model: str = Field(default="claude-sonnet-4-5", description="Nom du modèle Claude")
    timeout: int = Field(default=300, description="Timeout en secondes")
    claude_command: str = Field(default="claude", description="Commande CLI pour Claude")
    max_retries: int = Field(default=2, description="Nombre de tentatives")
    temperature: float = Field(default=1.0, description="Température (non utilisé par CLI)")
    base_url: Optional[str] = Field(default=None, description="Base URL (ignoré pour CLI)")

    _service: Optional[ClaudeCliService] = None

    class Config:
        """Configuration Pydantic."""
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize the Claude CLI chat model."""
        super().__init__(**kwargs)
        self._service = ClaudeCliService(
            timeout=self.timeout,
            claude_command=self.claude_command,
            max_retries=self.max_retries
        )

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "claude-cli"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "timeout": self.timeout,
            "claude_command": self.claude_command,
        }

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> tuple[Optional[str], str]:
        """
        Convertit une liste de messages LangChain en un prompt pour Claude CLI.

        Args:
            messages: Liste des messages du chat

        Returns:
            Tuple (system_prompt, user_prompt)
        """
        system_messages = []
        conversation = []

        for message in messages:
            if isinstance(message, SystemMessage):
                system_messages.append(message.content)
            elif isinstance(message, HumanMessage):
                conversation.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                conversation.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                conversation.append(f"{message.role}: {message.content}")
            else:
                # Fallback pour les types de messages inconnus
                conversation.append(f"Message: {message.content}")

        system_prompt = "\n\n".join(system_messages) if system_messages else None
        user_prompt = "\n\n".join(conversation)

        return system_prompt, user_prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Génère une réponse en utilisant Claude CLI.

        Args:
            messages: Liste des messages du chat
            stop: Séquences d'arrêt (non supporté par CLI)
            run_manager: Callback manager
            **kwargs: Arguments additionnels

        Returns:
            ChatResult contenant la réponse
        """
        if not self._service:
            self._service = ClaudeCliService(
                timeout=self.timeout,
                claude_command=self.claude_command,
                max_retries=self.max_retries
            )

        # Convertir les messages en prompt
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        try:
            # Exécuter Claude CLI
            response = self._service.execute(
                prompt=user_prompt,
                system_prompt=system_prompt,
                expect_json=kwargs.get('expect_json', False)
            )

            # Créer le message de réponse
            message = AIMessage(content=response)

            # Créer la génération
            generation = ChatGeneration(message=message)

            # Retourner le résultat
            return ChatResult(generations=[generation])

        except ClaudeCliError as e:
            # En cas d'erreur, retourner un message d'erreur
            error_message = AIMessage(
                content=f"Erreur lors de l'appel à Claude CLI: {str(e)}"
            )
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """
        Streaming non supporté pour Claude CLI.

        Note: Le CLI ne supporte pas nativement le streaming,
        donc on retourne la réponse complète en une fois.
        """
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        yield result.generations[0]

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Génère une réponse de manière asynchrone.

        Args:
            messages: Liste des messages du chat
            stop: Séquences d'arrêt (non supporté)
            run_manager: Callback manager
            **kwargs: Arguments additionnels

        Returns:
            ChatResult contenant la réponse
        """
        if not self._service:
            self._service = ClaudeCliService(
                timeout=self.timeout,
                claude_command=self.claude_command,
                max_retries=self.max_retries
            )

        # Convertir les messages en prompt
        system_prompt, user_prompt = self._convert_messages_to_prompt(messages)

        try:
            # Exécuter Claude CLI de manière asynchrone
            response = await self._service.execute_async(
                prompt=user_prompt,
                system_prompt=system_prompt,
                expect_json=kwargs.get('expect_json', False)
            )

            # Créer le message de réponse
            message = AIMessage(content=response)

            # Créer la génération
            generation = ChatGeneration(message=message)

            # Retourner le résultat
            return ChatResult(generations=[generation])

        except ClaudeCliError as e:
            # En cas d'erreur, retourner un message d'erreur
            error_message = AIMessage(
                content=f"Erreur lors de l'appel asynchrone à Claude CLI: {str(e)}"
            )
            generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[generation])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatClaudeCLI":
        """
        Bind tools to the model (limited support).

        Note: Claude CLI a un support limité pour les outils.
        Cette méthode est fournie pour compatibilité mais peut ne pas fonctionner
        comme attendu.
        """
        # Pour l'instant, on retourne simplement self
        # Une implémentation plus avancée pourrait formatter les tools dans le prompt
        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Claude CLI."""
        return {
            "model": self.model,
            "temperature": self.temperature,
        }


# Fonction helper pour créer une instance facilement
def create_claude_cli_chat(
    model: str = "claude-sonnet-4-5",
    timeout: int = 300,
    **kwargs
) -> ChatClaudeCLI:
    """
    Crée une instance de ChatClaudeCLI avec les paramètres donnés.

    Args:
        model: Nom du modèle Claude
        timeout: Timeout en secondes
        **kwargs: Arguments additionnels

    Returns:
        Instance de ChatClaudeCLI
    """
    return ChatClaudeCLI(model=model, timeout=timeout, **kwargs)
