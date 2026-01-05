"""
Claude CLI Service - Wrapper pour exécuter Claude Code via subprocess.

Ce module fournit une interface pour interagir avec Claude Code installé localement
via la ligne de commande, remplaçant les appels directs à l'API OpenAI.
"""

import subprocess
import json
import asyncio
import re
from typing import Optional, Dict, Any, List
from datetime import datetime


class ClaudeCliError(Exception):
    """Exception levée lors d'erreurs d'exécution de Claude CLI."""
    pass


class ClaudeCliService:
    """Service pour interagir avec Claude Code via CLI."""

    def __init__(
        self,
        timeout: int = 300,  # 5 minutes par défaut
        claude_command: str = "claude",
        max_retries: int = 2
    ):
        """
        Initialize Claude CLI Service.

        Args:
            timeout: Timeout en secondes pour chaque appel CLI
            claude_command: Commande pour invoquer Claude (défaut: "claude")
            max_retries: Nombre de tentatives en cas d'échec
        """
        self.timeout = timeout
        self.claude_command = claude_command
        self.max_retries = max_retries

    def _clean_response(self, raw_output: str) -> str:
        """
        Nettoie la réponse brute de Claude CLI.

        Args:
            raw_output: Sortie brute de la commande

        Returns:
            Réponse nettoyée
        """
        # Supprimer les lignes de debug, warnings, etc.
        lines = raw_output.split('\n')
        cleaned_lines = []

        for line in lines:
            # Ignorer les lignes vides et de debug
            if line.strip() and not any([
                line.startswith('['),
                line.startswith('DEBUG:'),
                line.startswith('INFO:'),
                line.startswith('WARNING:'),
                line.startswith('ERROR:'),
            ]):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Tente d'extraire un objet JSON de la réponse textuelle.

        Args:
            text: Texte contenant potentiellement du JSON

        Returns:
            Dictionnaire Python ou None si pas de JSON trouvé
        """
        # Chercher des patterns JSON
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested object
            r'\[.*?\]',  # Array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def execute(
        self,
        prompt: str,
        expect_json: bool = False,
        system_prompt: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> str:
        """
        Exécute une commande Claude CLI de manière synchrone.

        Args:
            prompt: Le prompt à envoyer à Claude
            expect_json: Si True, tente d'extraire et parser du JSON
            system_prompt: Prompt système optionnel
            additional_args: Arguments CLI additionnels

        Returns:
            Réponse de Claude (str ou dict si expect_json=True)

        Raises:
            ClaudeCliError: En cas d'erreur d'exécution
        """
        # Construire le prompt complet
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Construire la commande
        cmd = [self.claude_command]

        # Ajouter des arguments pour mode non-interactif
        if additional_args:
            cmd.extend(additional_args)

        # Ajouter le prompt
        cmd.append(full_prompt)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Exécuter la commande
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False  # Ne pas lever d'exception automatiquement
                )

                # Vérifier les erreurs
                if result.returncode != 0:
                    error_msg = result.stderr or "Erreur inconnue"
                    raise ClaudeCliError(
                        f"Claude CLI a retourné une erreur (code {result.returncode}): {error_msg}"
                    )

                # Nettoyer la sortie
                cleaned_output = self._clean_response(result.stdout)

                if not cleaned_output:
                    raise ClaudeCliError("Claude CLI n'a retourné aucune réponse")

                # Extraire JSON si demandé
                if expect_json:
                    json_data = self._extract_json(cleaned_output)
                    if json_data:
                        return json.dumps(json_data)
                    else:
                        # Retourner le texte brut si pas de JSON trouvé
                        return cleaned_output

                return cleaned_output

            except subprocess.TimeoutExpired as e:
                last_error = ClaudeCliError(
                    f"Timeout après {self.timeout}s lors de l'exécution de Claude CLI"
                )
                if attempt < self.max_retries:
                    continue

            except Exception as e:
                last_error = ClaudeCliError(f"Erreur inattendue: {str(e)}")
                if attempt < self.max_retries:
                    continue

        # Si on arrive ici, toutes les tentatives ont échoué
        raise last_error or ClaudeCliError("Échec après plusieurs tentatives")

    async def execute_async(
        self,
        prompt: str,
        expect_json: bool = False,
        system_prompt: Optional[str] = None,
        additional_args: Optional[List[str]] = None
    ) -> str:
        """
        Exécute une commande Claude CLI de manière asynchrone.

        Args:
            prompt: Le prompt à envoyer à Claude
            expect_json: Si True, tente d'extraire et parser du JSON
            system_prompt: Prompt système optionnel
            additional_args: Arguments CLI additionnels

        Returns:
            Réponse de Claude (str ou dict si expect_json=True)

        Raises:
            ClaudeCliError: En cas d'erreur d'exécution
        """
        # Construire le prompt complet
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Construire la commande
        cmd = [self.claude_command]

        if additional_args:
            cmd.extend(additional_args)

        cmd.append(full_prompt)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Créer le processus async
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Attendre avec timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    raise ClaudeCliError(
                        f"Timeout après {self.timeout}s lors de l'exécution de Claude CLI"
                    )

                # Décoder les sorties
                stdout_text = stdout.decode('utf-8')
                stderr_text = stderr.decode('utf-8')

                # Vérifier les erreurs
                if process.returncode != 0:
                    raise ClaudeCliError(
                        f"Claude CLI a retourné une erreur (code {process.returncode}): {stderr_text}"
                    )

                # Nettoyer la sortie
                cleaned_output = self._clean_response(stdout_text)

                if not cleaned_output:
                    raise ClaudeCliError("Claude CLI n'a retourné aucune réponse")

                # Extraire JSON si demandé
                if expect_json:
                    json_data = self._extract_json(cleaned_output)
                    if json_data:
                        return json.dumps(json_data)
                    else:
                        return cleaned_output

                return cleaned_output

            except ClaudeCliError:
                if attempt < self.max_retries:
                    await asyncio.sleep(1)  # Petit délai avant retry
                    continue
                raise

            except Exception as e:
                last_error = ClaudeCliError(f"Erreur inattendue: {str(e)}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue

        raise last_error or ClaudeCliError("Échec après plusieurs tentatives")


# Instance globale pour faciliter l'utilisation
_default_service = None

def get_default_service() -> ClaudeCliService:
    """Retourne l'instance par défaut du service Claude CLI."""
    global _default_service
    if _default_service is None:
        _default_service = ClaudeCliService()
    return _default_service


# Fonctions de commodité
def execute_claude(prompt: str, **kwargs) -> str:
    """Execute Claude CLI avec le prompt donné."""
    service = get_default_service()
    return service.execute(prompt, **kwargs)


async def execute_claude_async(prompt: str, **kwargs) -> str:
    """Execute Claude CLI de manière asynchrone."""
    service = get_default_service()
    return await service.execute_async(prompt, **kwargs)
