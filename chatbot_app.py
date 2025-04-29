import sys
import json
import requests
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import flet as ft
from llm_provider import LLMProvider, OllamaProvider  # Import from new file


# UI Components
class Message:
    def __init__(self, user_name: str, text: str):
        self.user_name = user_name
        self.text = text


class ChatMessage(ft.Container):
    def __init__(self, message: Message, page: ft.Page = None):
        super().__init__()
        self.animate = ft.animation.Animation(300, ft.AnimationCurve.EASE_OUT)
        self.padding = ft.padding.only(bottom=12)

        is_user = message.user_name == "You"

        # Create content area
        if is_user:
            # User message: Right-aligned bubble
            content_area = ft.Container(
                content=ft.Text(message.text, selectable=True, size=15, color=ft.Colors.BLACK87),
                bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.BLACK),
                border_radius=18,
                padding=ft.padding.symmetric(horizontal=16, vertical=10),
            )
            alignment = ft.MainAxisAlignment.END
            self.content = ft.Row([content_area], alignment=alignment)
        else:
            # Bot message: Left-aligned with markdown support
            content_area = ft.Container(
                content=ft.Markdown(
                    value=message.text, 
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                    code_theme=ft.MarkdownCodeTheme.GITHUB,
                    on_tap_link=lambda e: page.launch_url(e.data) if page else None,
                ),
                padding=ft.padding.only(left=20, right=16, top=4, bottom=4),
            )
            alignment = ft.MainAxisAlignment.START
            self.content = ft.Row([ft.Container(content=content_area, expand=True)], alignment=alignment)


class ModelSettingsPanel(ft.Container):
    def __init__(self, llm_provider: LLMProvider, on_save=None, page: ft.Page = None):
        self.llm_provider = llm_provider
        self.on_save = on_save
        self.page = page
        
        # Model selection dropdown
        self.model_combo = ft.Dropdown(
            label="Select Model",
            width=300,
            filled=True,
            hint_text="Select a model",
            helper_text="Choose from available local models",
            border_radius=10,
            expand=True,
        )
        
        # Temperature slider
        self.temperature_slider = ft.Slider(
            min=0,
            max=2.0,
            divisions=20,
            label="Temperature: {value}",
            value=0.7,  # Default temperature value
            width=300,
        )
        
        # Temperature label
        self.temp_label = ft.Text("Temperature: 0.7", size=14)
        
        # Update temperature label when slider changes
        self.temperature_slider.on_change = self._update_temperature_label
        
        # Model info display
        self.model_details = ft.Container(
            content=ft.Column([
                ft.Text("No model selected", italic=True, color=ft.Colors.BLACK45),
            ]),
            padding=10,
            border_radius=8,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.BLACK),
            visible=False,
        )
        
        # Ollama endpoint input
        self.endpoint_input = ft.TextField(
            label="Endpoint URL",
            value=self.llm_provider.host,
            width=300,
            border_radius=10,
            filled=True,
            prefix_icon=ft.Icons.LINK,
        )
        
        # Loading indicator
        self.loading = ft.ProgressRing(
            width=20, 
            height=20,
            stroke_width=2,
            visible=False,
        )
        
        # Set up onChange handler for model selection
        self.model_combo.on_change = self._on_model_selected
        
        # Call the parent class constructor with all the Container properties
        super().__init__(
            width=380,  # Fixed width for the panel
            bgcolor=ft.Colors.BLUE_GREY_50,
            padding=ft.padding.all(20),
            border_radius=ft.border_radius.only(
                top_left=10,
                bottom_left=10
            ),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.with_opacity(0.25, ft.Colors.BLACK),
                offset=ft.Offset(-2, 0),
            ),
            content=ft.Column([
                ft.Row([
                    ft.Text("Model Settings", size=20, weight="bold"),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        tooltip="Close",
                        on_click=self._handle_close,
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(height=1, color=ft.Colors.BLUE_GREY_200),
                ft.Container(height=10),  # Spacer
                ft.Text("Endpoint", size=16, weight="bold"),
                self.endpoint_input,
                ft.Container(height=20),  # Spacer
                ft.Text("Model Selection", size=16, weight="bold"),
                ft.Row([
                    self.model_combo,
                    ft.IconButton(
                        icon=ft.Icons.REFRESH,
                        tooltip="Refresh models",
                        on_click=self._refresh_models,
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.loading,
                ft.Container(height=10),  # Spacer
                self.model_details,
                ft.Container(height=20),  # Spacer
                ft.Text("Generation Settings", size=16, weight="bold"),
                self.temp_label,
                self.temperature_slider,
                ft.Container(height=20),  # Spacer
                ft.Row([
                    ft.ElevatedButton(
                        "Save",
                        icon=ft.Icons.SAVE,
                        on_click=self._handle_save,
                        style=ft.ButtonStyle(
                            color=ft.Colors.WHITE,
                            bgcolor=ft.Colors.BLUE,
                        )
                    ),
                    ft.OutlinedButton(
                        "Cancel",
                        on_click=self._handle_close,
                    ),
                ], alignment=ft.MainAxisAlignment.END),
            ], scroll=ft.ScrollMode.AUTO),
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_OUT),
            visible=False,  # Initially hidden - set this in the constructor
        )
    
    def _update_temperature_label(self, e):
        """Update the temperature label when slider changes"""
        temp = round(self.temperature_slider.value, 2)
        self.temp_label.value = f"Temperature: {temp}"
        self.temp_label.update()
    
    def toggle_visibility(self):
        """Toggle panel visibility"""
        self.visible = not self.visible
        if self.visible:
            # Populate models when panel becomes visible
            self._populate_models()
            # Set the temperature to the current value from the provider
            temp = getattr(self.llm_provider, "temperature", 0.7)
            self.temperature_slider.value = temp
            self._update_temperature_label(None)
        
    def _populate_models(self):
        """Fetch and populate the models dropdown"""
        self.loading.visible = True
        if self.page:
            self.page.update()
        
        try:
            # Get models from provider
            models = self.llm_provider.get_available_models()
            
            # Create dropdown options
            options = []
            
            # Check if models is empty
            if not models:
                options.append(ft.dropdown.Option("No models found", "No models found"))
            else:
                for model in models:
                    # The Ollama Python library returns each model with a 'model' field for the name
                    model_name = model.get('model', '')
                    
                    if not model_name:
                        continue
                        
                    # Extract details from the model object
                    details = model.get('details', {})
                    
                    # Build display text with useful information
                    display_parts = []
                    
                    # Add parameter size if available
                    param_size = details.get('parameter_size', '')
                    if param_size:
                        display_parts.append(param_size)
                        
                    # Add quantization level if available
                    quant_level = details.get('quantization_level', '')
                    if quant_level:
                        display_parts.append(quant_level)
                    
                    # Create final display text
                    if display_parts:
                        display_text = f"{model_name} ({', '.join(display_parts)})"
                    else:
                        display_text = model_name
                    
                    options.append(ft.dropdown.Option(model_name, display_text))
            
            # Update dropdown
            self.model_combo.options = options
            self.model_combo.disabled = False
            
            # Select current model
            self.model_combo.value = self.llm_provider.model_name if self.llm_provider.model_name else None
            
            # Update model details if a model is selected
            if self.model_combo.value:
                self._update_model_details(self.model_combo.value)
                
        except Exception as e:
            print(f"Error populating models: {e}")
            # Add a default "error" option
            self.model_combo.options = [ft.dropdown.Option("Error loading models", "Error loading models")]
            self.model_combo.disabled = False
        
        # Hide loading indicator
        self.loading.visible = False
        if self.page:
            self.page.update()
    
    def _on_model_selected(self, e):
        """Handle model selection change"""
        if e.data and e.data != "None":
            self._update_model_details(e.data)
    
    def _update_model_details(self, model_name: str):
        """Update the model details section with info about the selected model"""
        try:
            # Get model details from provider
            model_info = self.llm_provider.get_model_info(model_name)
            
            if not model_info or "error" in model_info:
                self.model_details.content = ft.Text(
                    f"Error fetching details for {model_name}",
                    color=ft.Colors.RED_400
                )
                self.model_details.visible = True
                return
            
            # Extract useful info
            details = model_info.get('details', {})
            model_family = details.get('family', 'Unknown')
            model_format = details.get('format', 'Unknown')
            parameter_size = details.get('parameter_size', 'Unknown')
            quantization = details.get('quantization_level', 'None')
            
            # Create a nice details display
            detail_rows = [
                ft.Row([
                    ft.Icon(ft.Icons.MODEL_TRAINING, size=16, color=ft.Colors.BLUE),
                    ft.Text(f"Model: ", weight="bold"),
                    ft.Text(model_name),
                ], spacing=5),
                ft.Row([
                    ft.Icon(ft.Icons.CATEGORY, size=16, color=ft.Colors.BLUE),
                    ft.Text(f"Family: ", weight="bold"),
                    ft.Text(model_family),
                ], spacing=5),
                ft.Row([
                    ft.Icon(ft.Icons.MEMORY, size=16, color=ft.Colors.BLUE),
                    ft.Text(f"Size: ", weight="bold"),
                    ft.Text(parameter_size),
                ], spacing=5),
            ]
            
            if quantization and quantization != "None":
                detail_rows.append(ft.Row([
                    ft.Icon(ft.Icons.COMPRESS, size=16, color=ft.Colors.BLUE),
                    ft.Text(f"Quantization: ", weight="bold"),
                    ft.Text(quantization),
                ], spacing=5))
                
            detail_rows.append(ft.Row([
                ft.Icon(ft.Icons.FILE_PRESENT, size=16, color=ft.Colors.BLUE),
                ft.Text(f"Format: ", weight="bold"),
                ft.Text(model_format),
            ], spacing=5))
            
            # Update the details container
            self.model_details.content = ft.Column(detail_rows, spacing=8)
            self.model_details.visible = True
            
        except Exception as e:
            print(f"Error fetching model details: {e}")
            self.model_details.content = ft.Text(
                f"Error: {str(e)}",
                color=ft.Colors.RED_400
            )
            self.model_details.visible = True
    
    def _refresh_models(self, e=None):
        """Refresh the model list"""
        # Update endpoint if changed
        endpoint = self.endpoint_input.value.strip()
        if endpoint and endpoint != self.llm_provider.host:
            self.llm_provider.host = endpoint
            
            # Create new client with updated host
            import ollama
            self.llm_provider.client = ollama.Client(host=endpoint)
        
        # Repopulate models
        self._populate_models()
    
    def _handle_save(self, e):
        """Save settings and close panel"""
        model_name = self.model_combo.value
        endpoint = self.endpoint_input.value.strip()
        temperature = self.temperature_slider.value
        
        # Update provider settings
        if endpoint != self.llm_provider.host:
            self.llm_provider.host = endpoint
            # Update ollama client configuration
            import ollama
            self.llm_provider.client = ollama.Client(host=endpoint)
        
        # Update model
        if model_name and model_name != "No models found":
            success = self.llm_provider.set_model(model_name)
            if not success and self.page:
                # Create a snackbar and show it (correct Flet approach)
                snack = ft.SnackBar(
                    content=ft.Text(f"Failed to set model to {model_name}"),
                    action="Dismiss"
                )
                self.page.add(snack)
                snack.open = True
                self.page.update()
                return
        
        # Update temperature
        self.llm_provider.temperature = temperature
        
        # Call the save callback
        if self.on_save:
            self.on_save(model_name, endpoint, temperature)
        
        # Close panel
        self._handle_close(e)
    
    def _handle_close(self, e):
        """Close the panel"""
        self.visible = False
        if self.page:
            self.page.update()


class ChatApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.llm_provider = OllamaProvider()
        # Create an event loop for async operations
        self.loop = asyncio.new_event_loop()
        self.setup_page()
        self.build_ui()
    
    def setup_page(self):
        self.page.title = "Chatbot Application"
        self.page.window_width = 900
        self.page.window_height = 700
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.bgcolor = ft.Colors.with_opacity(0.98, ft.Colors.WHITE)
        self.page.fonts = {
            "SF Pro": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        }
        self.page.theme = ft.Theme(font_family="SF Pro, Inter, Helvetica, Arial, sans-serif")
        self.page.window_min_width = 450
        self.page.window_min_height = 600
        self.page.window_maximizable = True
        self.page.window_resizable = True
        self.page.scroll = "auto"
        # Set adaptive to true for responsive layout
        self.page.adaptive = True
    
    def build_ui(self):
        # Create chat list with improved padding and spacing
        self.chat_display = ft.ListView(
            expand=True,
            spacing=10,
            auto_scroll=True,
            padding=ft.padding.only(left=20, right=20, bottom=10),
        )
        
        # Add welcome message
        welcome_message = Message("Bot", "Welcome to the Chatbot Application! Type a message to begin.")
        self.chat_display.controls.append(ChatMessage(welcome_message, self.page))
        
        # Create typing indicator
        self.typing_indicator = ft.Container(
            content=ft.Row(
                [
                    ft.Container(
                        content=ft.ProgressRing(width=16, height=16, stroke_width=2),
                        margin=ft.margin.only(right=8)
                    ),
                    ft.Text("Assistant is thinking...", size=14, color=ft.Colors.BLACK54),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            margin=ft.margin.only(left=20, bottom=12),
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
            visible=False
        )
        
        # Create input field with improved styling
        self.chat_input = ft.TextField(
            hint_text="Type your message here...",
            border_radius=25,
            min_lines=1, 
            max_lines=5,
            filled=True,
            expand=True,
            on_submit=self.send_message,
            shift_enter=True,
            text_size=15,
            content_padding=ft.padding.only(left=20, right=20, top=14, bottom=14),
            cursor_color=ft.Colors.BLUE,
            focused_border_color=ft.Colors.BLUE,
        )
        
        # Create send button with improved styling
        send_button = ft.IconButton(
            icon=ft.Icons.SEND_ROUNDED,
            icon_color=ft.Colors.BLUE,
            icon_size=22,
            tooltip="Send message",
            on_click=self.send_message,
        )
        
        # Create model settings panel
        self.model_settings_panel = ModelSettingsPanel(
            self.llm_provider,
            on_save=self._on_settings_saved,
            page=self.page
        )
        
        # Create model settings button with improved styling
        model_button = ft.ElevatedButton(
            "Model Settings",
            icon=ft.Icons.SETTINGS,
            on_click=self._toggle_model_settings,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE,
            )
        )
        
        # Create sidebar toggle button
        sidebar_toggle = ft.IconButton(
            icon=ft.Icons.MENU,
            icon_size=24,
            on_click=self.toggle_sidebar,
            icon_color=ft.Colors.WHITE,
        )
        
        # Create top bar with improved styling
        top_bar = ft.Container(
            content=ft.Row(
                [
                    sidebar_toggle,
                    ft.Container(width=10),  # Spacer
                    ft.Text("Chatbot Application", size=20, weight="bold", color=ft.Colors.WHITE),
                    # Replace Spacer with Container that expands
                    ft.Container(expand=True),
                    model_button,
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            bgcolor=ft.Colors.BLUE_700,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=5,
                color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                offset=ft.Offset(0, 1),
            ),
        )
        
        # Create sidebar with improved styling
        self.sidebar = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text("Menu", size=18, weight="bold"),
                        padding=10,
                    ),
                    ft.Divider(),
                    ft.TextButton("Option 1", on_click=lambda _: None),
                    ft.TextButton("Option 2", on_click=lambda _: None),
                    ft.TextButton("Option 3", on_click=lambda _: None),
                    ft.TextButton("Option 4", on_click=lambda _: None),
                    ft.TextButton("Option 5", on_click=lambda _: None),
                ],
                spacing=5,
            ),
            width=0,  # Initially hidden
            bgcolor=ft.Colors.BLUE_GREY_50,
            animate=ft.animation.Animation(300, "easeOut"),
        )
        
        # Create input area with improved styling and fixed at bottom
        input_area = ft.Container(
            content=ft.Row(
                [self.chat_input, send_button],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.symmetric(horizontal=16, vertical=12),
            bgcolor=ft.Colors.WHITE,
            border=ft.border.only(top=ft.border.BorderSide(1, ft.Colors.BLACK12)),
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
                offset=ft.Offset(0, -2),
            ),
        )
        
        # Center the main content
        main_content_container = ft.Container(
            content=ft.Column(
                [
                    top_bar,
                    ft.Container(
                        content=self.chat_display,
                        expand=True,
                        bgcolor=ft.Colors.WHITE,
                    ),
                    input_area,
                ],
                spacing=0,
                expand=True,
            ),
            expand=True,
        )
        
        # Wrap in a row for horizontal centering with max width
        centered_content = ft.Row(
            [
                ft.Container(
                    content=main_content_container,
                    width=800,  # Set max width for the chat interface
                    expand=True,  # But allow it to fill available space up to the max width
                )
            ],
            expand=True,
            alignment=ft.MainAxisAlignment.CENTER,  # Center horizontally
        )
        
        # Create main layout with centered content
        self.page.add(
            ft.Stack([
                centered_content,
                # Settings panel overlay
                ft.Row(
                    [
                        ft.Container(expand=True),  # Push panel to the right
                        self.model_settings_panel,
                    ],
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
            ], expand=True)
        )
        
    def toggle_sidebar(self, e):
        self.sidebar.width = 250 if self.sidebar.width == 0 else 0
        self.sidebar.update()
        
    def _toggle_model_settings(self, e):
        """Toggle the model settings panel"""
        self.model_settings_panel.toggle_visibility()
        self.page.update()
    
    def _on_settings_saved(self, model_name, endpoint, temperature):
        """Handle saved settings"""
        self.chat_display.controls.append(
            ChatMessage(Message("Bot", f"*Settings updated:* Model: {model_name}, Temperature: {temperature}"), self.page)
        )
        self.page.update()
        
    async def process_message(self, user_message: str):
        self.typing_indicator.visible = True
        self.chat_display.controls.append(self.typing_indicator)
        self.page.update()
        
        try:
            # Use chat endpoint with temperature
            messages = [{"role": "user", "content": user_message}]
            response = self.llm_provider.generate_chat(
                messages, 
                temperature=getattr(self.llm_provider, "temperature", 0.7)
            )
            
            # Hide typing indicator
            self.typing_indicator.visible = False
            if self.typing_indicator in self.chat_display.controls:
                self.chat_display.controls.remove(self.typing_indicator)
            
            # Display bot response
            bot_message = Message("Bot", response)
            self.chat_display.controls.append(ChatMessage(bot_message, self.page))
            self.page.update()
            
        except Exception as e:
            # Hide typing indicator
            self.typing_indicator.visible = False
            if self.typing_indicator in self.chat_display.controls:
                self.chat_display.controls.remove(self.typing_indicator)
            
            # Display error message
            error_message = Message("Bot", f"Error: {str(e)}")
            self.chat_display.controls.append(ChatMessage(error_message, self.page))
            self.page.update()
    
    def _run_async_in_thread(self, coro):
        """Helper function to run async code in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(coro)
            
    def send_message(self, e):
        user_message = self.chat_input.value.strip()
        if not user_message:
            return
            
        # Display user message
        user_msg = Message("You", user_message)
        self.chat_display.controls.append(ChatMessage(user_msg, self.page))
        
        # Clear input
        self.chat_input.value = ""
        self.chat_input.focus()
        self.page.update()
        
        # Process message in a separate thread to avoid blocking the UI
        thread = threading.Thread(
            target=self._run_async_in_thread,
            args=(self.process_message(user_message),)
        )
        thread.daemon = True
        thread.start()


# Create async main to properly handle async operations
async def main(page: ft.Page):
    app = ChatApp(page)


if __name__ == '__main__':
    ft.app(target=main)