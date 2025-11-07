"""Generate synthetic invoice data using LangChain and LangGraph"""

import concurrent.futures
import csv
import json
import logging
import operator
import os
import random
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional, Tuple
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from chat_model import LLMManager
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# List of expense categories
EXPENSES = [
    "Groceries",
    "Housing",
    "Dining/Restaurant",
    "Transportation",
    "Healthcare",
    "Personal Care",
    "Entertainment",
    "Clothing",
    "Education",
    "Electronics",
    "Home Maintenance",
    "Pet Expenses",
    "Gifts",
    "Personal Services",
    "Office Supplies",
    "Equipment Purchase",
    "Professional Services",
    "Software/Subscriptions",
    "Marketing/Advertising",
    "Travel Expenses",
    "Training/Education",
    "Consulting Fees",
    "Rent/Facilities",
    "Utilities",
    "Insurance",
    "Legal Services",
    "Accounting/Financial Services",
    "Research & Development",
    "Inventory",
    "Shipping/Logistics",
    "Vehicle Expenses",
    "Communication Services",
    "Freelance/Contract Work",
    "Rental Income",
    "Reimbursable Expenses",
    "Investment-Related",
    "Tax-Related Expenses",
]

# Define frequency patterns for different expense categories
EXPENSE_FREQUENCIES = {
    # Daily/near-daily expenses
    "Groceries": {"days": 1, "variance": 2},  # Every 1-3 days
    "Dining/Restaurant": {"days": 3, "variance": 2},  # Every 3-5 days
    "Transportation": {"days": 1, "variance": 1},  # Every 1-2 days
    # Weekly expenses
    "Personal Care": {"days": 7, "variance": 3},  # Every 7-10 days
    "Entertainment": {"days": 7, "variance": 4},  # Every 7-11 days
    # Bi-weekly expenses
    "Clothing": {"days": 14, "variance": 5},  # Every 14-19 days
    "Home Maintenance": {"days": 14, "variance": 7},  # Every 14-21 days
    # Monthly expenses
    "Housing": {"days": 30, "variance": 2},  # Every 30-32 days
    "Utilities": {"days": 30, "variance": 3},  # Every 30-33 days
    "Insurance": {"days": 30, "variance": 0},  # Exactly every 30 days
    "Software/Subscriptions": {"days": 30, "variance": 0},  # Exactly every 30 days
    # Quarterly expenses
    "Electronics": {"days": 90, "variance": 15},  # Every 90-105 days
    "Professional Services": {"days": 90, "variance": 10},  # Every 90-100 days
    # Default (bi-weekly)
    "DEFAULT": {"days": 14, "variance": 7},  # Every 14-21 days
}


# State definitions for LangGraph
class ExpenseState(TypedDict):
    expense_category: str
    samples_to_generate: int  # This will now be used as a maximum
    samples_generated: int
    output_file: str
    results: List[Dict]
    errors: List[str]
    api_calls: int
    current_date: datetime
    frequency: Dict
    end_date: datetime  # Add this to track the end date


class BatchState(TypedDict):
    expenses: List[str]
    num_samples_per_category: int
    output_dir: str
    completed_expenses: Annotated[List[str], operator.add]
    failed_expenses: Annotated[List[str], operator.add]
    total_api_calls: Annotated[int, operator.add]


class SyntheticDataGenerator:
    """Generate synthetic invoice data using LangChain and LangGraph"""

    def __init__(self):
        self.llm_manager = LLMManager.get_instance()
        self.client = self.llm_manager.get_llm()
        self.expense_graph = self._create_expense_graph()
        self.batch_graph = self._create_batch_graph()

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(
            f"Initialized SyntheticDataGenerator with output directory: {config.output_dir}"
        )

    def _create_expense_graph(self) -> StateGraph:
        """Create the graph for processing a single expense category"""
        builder = StateGraph(ExpenseState)

        # Add nodes
        builder.add_node("initialize", self._initialize_expense)
        builder.add_node("generate_sample", self._generate_sample)
        builder.add_node("save_results", self._save_results)

        # Add edges
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate_sample")

        # Conditional edge: either generate more samples or finish
        builder.add_conditional_edges(
            "generate_sample",
            self._should_continue_generating,
            {"continue": "generate_sample", "finish": "save_results"},
        )

        builder.add_edge("save_results", END)

        logger.debug("Created expense graph")
        return builder.compile()

    def _create_batch_graph(self) -> StateGraph:
        """Create the graph for processing multiple expense categories"""
        builder = StateGraph(BatchState)

        # Add nodes
        builder.add_node("initialize_batch", self._initialize_batch)
        builder.add_node("process_expense", self._process_expense)

        # Add edges
        builder.add_edge(START, "initialize_batch")

        # Create parallel tasks for each expense category
        builder.add_conditional_edges(
            "initialize_batch", self._create_expense_tasks, ["process_expense"]
        )

        builder.add_edge("process_expense", END)

        logger.debug("Created batch graph")
        return builder.compile()

    def _initialize_expense(self, state: ExpenseState) -> ExpenseState:
        """Initialize the state for processing an expense category"""
        state["samples_generated"] = 0
        state["results"] = []
        state["errors"] = []
        state["api_calls"] = 0
        state["output_file"] = os.path.join(
            config.output_dir, f"{state['expense_category'].replace('/', '_')}.csv"
        )

        # Initialize date generation
        # Start from January 1, 2 0 2 4
        state["current_date"] = datetime(2023, 1, 1)

        # End date (December 31, 2 0 2 4)
        state["end_date"] = datetime(2024, 12, 31)

        # Get frequency pattern for this category
        frequency = EXPENSE_FREQUENCIES.get(
            state["expense_category"], EXPENSE_FREQUENCIES["DEFAULT"]
        )
        state["frequency"] = frequency

        logger.debug(
            f"Initialized expense state for {state['expense_category']} with frequency {frequency}"
        )
        return state

    def _generate_sample(self, state: ExpenseState) -> ExpenseState:
        """Generate a single sample for the expense category"""
        try:
            # Generate date for this sample
            if state["samples_generated"] > 0:
                # Calculate next date based on frequency pattern
                frequency = state["frequency"]
                days_to_add = frequency["days"] + random.randint(
                    0, frequency["variance"]
                )
                state["current_date"] += timedelta(days=days_to_add)

            invoice_date = state["current_date"].strftime("%Y-%m-%d")

            # Create a prompt that includes the date
            prompt = self._create_prompt(state["expense_category"], invoice_date)

            # Track API calls
            state["api_calls"] += 1

            # Using the same approach as in data_extraction_agent.py
            system_prompt = (
                "You are an expert at generating synthetic data for invoices."
            )
            response = self.client.invoke(
                [
                    ("system", system_prompt),
                    ("user", prompt),
                ]
            )

            # Parse the JSON response
            data_dict = self._parse_json_response(response.content)

            if data_dict:
                # Ensure the invoice date matches our generated date
                data_dict["invoice_date"] = invoice_date

                # Calculate due date (typically 15-30 days after invoice date)
                due_days = random.randint(15, 30)
                due_date = (state["current_date"] + timedelta(days=due_days)).strftime(
                    "%Y-%m-%d"
                )
                data_dict["due_date"] = due_date

                state["results"].append(data_dict)
                state["samples_generated"] += 1
                logger.info(
                    f"Generated sample {state['samples_generated']} for {state['expense_category']} with date {invoice_date}"
                )
            else:
                state["errors"].append(
                    f"Failed to parse sample {state['samples_generated'] + 1}"
                )
                logger.warning(
                    f"Failed to parse sample {state['samples_generated'] + 1} for {state['expense_category']}"
                )

        except Exception as e:
            state["errors"].append(
                f"Error generating sample {state['samples_generated'] + 1}: {str(e)}"
            )
            logger.error(
                f"Error generating sample for {state['expense_category']}: {str(e)}"
            )

        return state

    def _should_continue_generating(self, state: ExpenseState) -> str:
        """Determine if we should continue generating samples"""
        # Stop if we've reached the maximum number of samples
        if (
            state["samples_to_generate"] > 0
            and state["samples_generated"] >= state["samples_to_generate"]
        ):
            return "finish"

        # Stop if we've reached or passed the end date
        if state["current_date"] >= state["end_date"]:
            return "finish"

        return "continue"

    def _save_results(self, state: ExpenseState) -> ExpenseState:
        """Save the generated samples to a CSV file"""
        if not state["results"]:
            logger.warning(f"No results to save for {state['expense_category']}")
            return state

        try:
            # Get fieldnames from the first result
            fieldnames = list(state["results"][0].keys())

            # Write to CSV
            with open(
                state["output_file"], "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in state["results"]:
                    writer.writerow(result)

            logger.info(
                f"Saved {len(state['results'])} samples to {state['output_file']}"
            )

        except Exception as e:
            state["errors"].append(f"Error saving results: {str(e)}")
            logger.error(
                f"Error saving results for {state['expense_category']}: {str(e)}"
            )

        return state

    def _initialize_batch(self, state: BatchState) -> BatchState:
        """Initialize the state for batch processing"""
        state["completed_expenses"] = []
        state["failed_expenses"] = []
        state["total_api_calls"] = 0
        logger.debug("Initialized batch state")
        return state

    def _create_expense_tasks(self, state: BatchState) -> List[Send]:
        """Create parallel tasks for each expense category"""
        logger.info(f"Creating tasks for {len(state['expenses'])} expense categories")
        return [
            Send(
                "process_expense",
                ExpenseState(
                    expense_category=expense,
                    samples_to_generate=state["num_samples_per_category"],
                    samples_generated=0,
                    output_file="",
                    results=[],
                    errors=[],
                    api_calls=0,
                    current_date=datetime(2023, 1, 1),
                    frequency=EXPENSE_FREQUENCIES.get(
                        expense, EXPENSE_FREQUENCIES["DEFAULT"]
                    ),
                ),
            )
            for expense in state["expenses"]
        ]

    def _generate_sample_for_date(self, category: str, date: datetime) -> Tuple[Dict, List[str], int]:
        """Generate a single sample for a specific date and category"""
        try:
            invoice_date = date.strftime("%Y-%m-%d")
            prompt = self._create_prompt(category, invoice_date)
            
            # Using the same approach as in data_extraction_agent.py
            system_prompt = "You are an expert at generating synthetic data for invoices."
            
            # Track API call
            api_calls = 1
            
            response = self.client.invoke([
                ("system", system_prompt),
                ("user", prompt),
            ])
            
            # Parse the JSON response
            data_dict = self._parse_json_response(response.content)
            
            if data_dict:
                # Ensure the invoice date matches our generated date
                data_dict["invoice_date"] = invoice_date
                
                # Calculate due date (typically 15-30 days after invoice date)
                due_days = random.randint(15, 30)
                due_date = (date + timedelta(days=due_days)).strftime("%Y-%m-%d")
                data_dict["due_date"] = due_date
                
                return data_dict, [], api_calls  # Success: return data, no errors, API calls
            else:
                return None, ["Failed to parse response"], api_calls  # Failure: no data, error message, API calls
                
        except Exception as e:
            return None, [f"Error generating sample: {str(e)}"], 1  # Exception: no data, error message, 1 API call

    def _process_expense(self, state: ExpenseState) -> Dict:
        """Process a single expense category with parallel sample generation"""
        logger.info(f"Processing expense category: {state['expense_category']}")

        # Initialize
        category = state["expense_category"]
        frequency = EXPENSE_FREQUENCIES.get(category, EXPENSE_FREQUENCIES["DEFAULT"])
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Calculate all the dates we need to generate samples for
        dates = []
        current_date = start_date

        # Either generate up to max_samples or until end_date is reached
        max_samples = state["samples_to_generate"]

        while current_date <= end_date and (
            max_samples <= 0 or len(dates) < max_samples
        ):
            dates.append(current_date)
            # Calculate next date based on frequency pattern
            days_to_add = frequency["days"] + random.randint(0, frequency["variance"])
            current_date += timedelta(days=days_to_add)

        # Limit to max_samples if specified
        if max_samples > 0 and len(dates) > max_samples:
            dates = dates[:max_samples]

        logger.info(f"Will generate {len(dates)} samples for {category}")

        # Generate samples in parallel
        results = []
        errors = []
        api_calls = 0

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(self._generate_sample_for_date, category, date): date 
                for date in dates
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    data, sample_errors, calls = future.result()
                    api_calls += calls

                    if data:
                        results.append(data)
                        logger.info(
                            f"Generated sample for {category} with date {date.strftime('%Y-%m-%d')}"
                        )

                    if sample_errors:
                        errors.extend(sample_errors)
                        logger.warning(
                            f"Errors for {category} with date {date.strftime('%Y-%m-%d')}: {sample_errors}"
                        )

                except Exception as e:
                    errors.append(f"Exception processing result: {str(e)}")
                    logger.error(
                        f"Exception for {category} with date {date.strftime('%Y-%m-%d')}: {str(e)}"
                    )

        # Save results to CSV
        if results:
            output_file = os.path.join(
                config.output_dir, f"{category.replace('/', '_')}.csv"
            )

            try:
                # Get fieldnames from the first result
                fieldnames = list(results[0].keys())

                # Write to CSV
                with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                    for result in results:
                        writer.writerow(result)

                logger.info(f"Saved {len(results)} samples to {output_file}")

            except Exception as e:
                errors.append(f"Error saving results: {str(e)}")
                logger.error(f"Error saving results for {category}: {str(e)}")

        # Return a simplified result with API call tracking
        return {
            f"result_{category}": {
                "expense_category": category,
                "samples_generated": len(results),
                "success": len(errors) == 0,
                "errors": errors,
                "api_calls": api_calls,
            }
        }

    def _create_prompt(self, expense_category: str, invoice_date: str = None) -> str:
        """Create the prompt for generating synthetic data"""
        date_instruction = ""
        if invoice_date:
            date_instruction = f" Use exactly this invoice date: {invoice_date}."

        return f"""You are an expert at generating synthetic data for attributes from invoices. For context, you are to generate data for an upper-middle class Indian living in Chennai, India. Generate a set of data for the \"{expense_category}\" category.{date_instruction} Ensure that all prices add up correctly. The following are the invoice attributes. If something is not applicable, use \"NULL\".

Output Format (JSON-style dictionary):
{{
    \"invoice_number\": \"The unique identifier for this invoice. (TEXT)\",
    \"invoice_date\": \"The date the invoice was issued (format: YYYY-MM-DD). (DATE)\",
    \"due_date\": \"The date by which payment is expected (format: YYYY-MM-DD). (DATE)\",
    \"seller_information\": \"Full name, address, and contact details of the seller. (TEXT)\",
    \"buyer_information\": \"Full name, address, and contact details of the buyer. (TEXT)\",
    \"purchase_order_number\": \"The buyer's purchase order number, if available. (TEXT)\",
    \"products_services\": \"Comma-separated list of all items or services billed. Do not include services like shipping. (TEXT)\",
    \"quantities\": \"Comma-separated list of quantities for each item, in the same order as the products/services. Do not include commas in each quantity itself. (INTEGER)\",
    \"unit_prices\": \"Comma-separated list of unit prices for each item, in the same order as the products/services. Do not include commas in each unit price itself. (NUMERIC)\",
    \"subtotal\": \"The sum of all line items before taxes and discounts. Do not include any commas in the subtotal. (NUMERIC)\",
    \"service_charges\": \"Any additional charges that may be applied. Do not include shipping costs here. Do not include any commas in the service charge. (NUMERIC)\",
    \"net_total\": \"Sum of subtotal and service charges. Do not include any commas in the net total. (NUMERIC)\",
    \"discount\": \"Any discounts applied to the invoice. Do not include any commas in the discount. (TEXT)\",
    \"tax\": \"The total amount of tax charged. Do not include any commas in the tax. (NUMERIC)\",
    \"tax_rate\": \"The percentage rate at which tax is charged. Do not include any commas in the tax rate. (TEXT)\",
    \"shipping_costs\": \"Any shipping or delivery charges. Do not include any commas in the shipping costs. (NUMERIC)\",
    \"grand_total\": \"The final amount to be paid, including all taxes and fees. Do not include any commas in the grand total. (NUMERIC)\",
    \"currency\": \"The currency in which the invoice is issued (INR, USD, SGD, AUD, etc). (TEXT)\",
    \"payment_terms\": \"The terms of payment (e.g., \\\"Net 30\\\", \\\"Due on Receipt\\\"). (TEXT)\",
    \"payment_method\": \"Accepted or preferred payment methods. (TEXT)\",
    \"bank_information\": \"Seller's bank details for payment, if provided. (TEXT)\",
    \"invoice_notes\": \"Any additional notes or terms on the invoice. (TEXT)\",
    \"shipping_address\": \"The delivery address. (TEXT)\",
    \"billing_address\": \"The billing address. (TEXT)\"
}}"""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse the JSON response from the model, handling potential formatting issues"""
        if not response_text:
            return None

        # Clean up the response text
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            # Try to parse the JSON
            data = json.loads(cleaned_text)

            # Clean up newlines in all string values
            cleaned_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    # Replace newlines with spaces and clean up multiple spaces
                    cleaned_value = value.replace("\n", " ").replace("\r", " ")
                    # Clean up multiple spaces that might have been created
                    cleaned_value = " ".join(cleaned_value.split())
                    cleaned_data[key] = cleaned_value
                else:
                    cleaned_data[key] = value

            return cleaned_data

        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the text
            try:
                start_idx = cleaned_text.find("{")
                end_idx = cleaned_text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = cleaned_text[start_idx:end_idx]
                    data = json.loads(json_str)

                    # Clean up newlines in all string values
                    cleaned_data = {}
                    for key, value in data.items():
                        if isinstance(value, str):
                            # Replace newlines with spaces and clean up multiple spaces
                            cleaned_value = value.replace("\n", " ").replace("\r", " ")
                            # Clean up multiple spaces that might have been created
                            cleaned_value = " ".join(cleaned_value.split())
                            cleaned_data[key] = cleaned_value
                        else:
                            cleaned_data[key] = value

                    return cleaned_data
            except:
                return None

    def generate_for_categories(self, categories: List[str], samples_per_category: int = None) -> List[Dict]:
        """Generate synthetic data for multiple expense categories using direct parallel processing"""
        if samples_per_category is None:
            samples_per_category = config.num_samples_per_category
        
        logger.info(f"Starting generation for {len(categories)} categories with {samples_per_category} samples each")
        
        results = []
        total_api_calls = 0
        
        # Process each category directly
        for category in categories:
            logger.info(f"Processing category: {category}")
            
            # Create a state for this category
            state = ExpenseState(
                expense_category=category,
                samples_to_generate=samples_per_category,
                samples_generated=0,
                output_file="",
                results=[],
                errors=[],
                api_calls=0,
                current_date=datetime(2023, 1, 1),
                frequency=EXPENSE_FREQUENCIES.get(category, EXPENSE_FREQUENCIES["DEFAULT"]),
                end_date=datetime(2024, 12, 31)
            )
            
            # Process this category
            result = self._process_expense(state)
            
            # Extract the result for this category
            category_result = result.get(f"result_{category}")
            if category_result:
                results.append(category_result)
                total_api_calls += category_result.get("api_calls", 0)
                logger.info(f"Completed {category}: {category_result.get('samples_generated', 0)} samples, {category_result.get('api_calls', 0)} API calls")
        
        logger.info(f"Total API calls made: {total_api_calls}")
        return results

    def generate_all(self, samples_per_category: int = None) -> Dict:
        """Generate synthetic data for all expense categories"""
        return self.generate_for_categories(EXPENSES, samples_per_category)

    def save_graph_visualization(self, output_dir: str = "graph_visualizations"):
        """Save visualizations of the LangGraph graphs"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save expense graph visualization
            expense_graph_path = os.path.join(output_dir, "expense_graph.png")
            self.expense_graph.get_graph().draw_png(expense_graph_path)
            logger.info(f"Saved expense graph visualization to {expense_graph_path}")
            
            # Save batch graph visualization
            batch_graph_path = os.path.join(output_dir, "batch_graph.png")
            self.batch_graph.get_graph().draw_png(batch_graph_path)
            logger.info(f"Saved batch graph visualization to {batch_graph_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save graph visualizations: {str(e)}")
            return False


def main():
    """Main entry point for the script"""
    generator = SyntheticDataGenerator()
    
    # Save graph visualizations
    generator.save_graph_visualization()
    
    # For testing, use just the first 2 categories
    test_categories = EXPENSES[:1]
    
    print(f"Generating synthetic data for {len(test_categories)} expense categories...")
    results = generator.generate_for_categories(test_categories)
    
    # Print summary
    print("\nGeneration complete!")
    print(f"Categories processed: {len(results)}")
    
    total_api_calls = sum(item.get("api_calls", 0) for item in results)
    print(f"Total API calls made: {total_api_calls}")
    
    for result in results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{status}: {result['expense_category']} - {result['samples_generated']} samples ({result.get('api_calls', 0)} API calls)")
        
        if not result["success"]:
            print(f"  Errors: {', '.join(result['errors'])}")


if __name__ == "__main__":
    main()
