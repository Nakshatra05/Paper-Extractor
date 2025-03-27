import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import openai
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define schema using Pydantic for validation
class NetData(BaseModel):
    Study_type: Optional[str] = Field(None, description="One of: Hut trial, lab based bioassay, or village trial")
    Net_type: str = Field(..., description="The type or brand name of the insecticide-treated net")
    Insecticide: Optional[str] = Field(None, description="The insecticide active ingredient in the net")
    Num_washes: Optional[int] = Field(None, description="Number of times the net was washed")
    Source: Optional[str] = Field(None, description="Mosquito source: 'Wild' or 'Lab'")
    Mosquito_spp: Optional[str] = Field(None, description="Mosquito species tested with the net")
    Site: Optional[str] = Field(None, description="Specific geographic information")
    Start_date: Optional[str] = Field(None, description="Start date in YYYY-MM format")
    End_date: Optional[str] = Field(None, description="End date in YYYY-MM format")
    Time_elapsed: Optional[float] = Field(None, description="Duration in months")

def create_extraction_prompt(paper_text: str, net_types: List[str]) -> str:
    """
    Create a prompt for the LLM to extract structured data from the paper text.
    
    Args:
        paper_text: The full text of the research paper
        net_types: List of net types to extract data for
        
    Returns:
        A formatted prompt string
    """
    # Convert the list of net types to a formatted string
    net_types_str = ", ".join([f"'{net}'" for net in net_types])
    
    prompt = f"""
Extract structured information from this research paper about insecticide-treated nets for malaria control.
For each of the following net types: {net_types_str}, extract the data into a separate JSON object.

Extract these fields for each net type:
- Study_type: One of "Hut trial", "lab based bioassay", or "village trial"
- Net_type: The type or brand name of the net (use the exact name from the list provided)
- Insecticide: The name of the insecticide active ingredient in the net
- Num_washes: The number of times the net was washed (as an integer)
- Source: Whether mosquitoes were from the wild or lab strain, answer only "Wild" or "Lab"
- Mosquito_spp: The mosquito species tested
- Site: Specific geographic location information
- Start_date: Study start date in YYYY-MM format
- End_date: Study end date in YYYY-MM format
- Time_elapsed: Duration of the study in months (as a number)

Output a JSON list where each object represents data for one net type. Only include fields where information is clearly provided in the paper.
For fields where information is not available, use null.

Paper Text:
{paper_text}

Your response should be a valid JSON array of objects, with one object per net type.
"""
    return prompt

def extract_data_with_openai(paper_text: str, net_types: List[str], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract data from paper text using OpenAI.
    
    Args:
        paper_text: Full paper text
        net_types: List of net types to extract data for
        api_key: OpenAI API key
        
    Returns:
        List of dictionaries containing extracted data for each net type
    """
    try:
        # Set API key if provided
        if api_key:
            openai.api_key = api_key
        
        prompt = create_extraction_prompt(paper_text, net_types)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k", # Using 16k model for longer paper texts
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from research papers about insecticide-treated nets for malaria control."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        
        # Parse the JSON
        extracted_data = json.loads(result_text)
        
        # Ensure we have a list
        if not isinstance(extracted_data, list):
            extracted_data = [extracted_data]
            
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in OpenAI extraction: {str(e)}")
        return []

def extract_data_with_llamaindex(paper_text: str, net_types: List[str], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract data from paper text using LlamaIndex.
    
    Args:
        paper_text: Full paper text
        net_types: List of net types to extract data for
        api_key: API key for OpenAI (used by LlamaIndex)
        
    Returns:
        List of dictionaries containing extracted data for each net type
    """
    try:
        from llama_index.program import OpenAIPydanticProgram
        import os
        
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Define a Pydantic model to hold all net types
        from pydantic import BaseModel, Field
        from typing import List as PyList
        
        class AllNetData(BaseModel):
            net_data: PyList[NetData] = Field(..., description="List of data for each net type")
        
        # Create the extraction program
        prompt_template = """
        Extract structured information from this research paper about insecticide-treated nets for malaria control.
        For each of the following net types: {net_types}, extract the data and return as a list of objects.
        
        Only include fields where information is clearly provided in the paper.
        
        Paper Text:
        {paper_text}
        """
        
        program = OpenAIPydanticProgram.from_defaults(
            output_cls=AllNetData,
            prompt_template_str=prompt_template,
            verbose=True,
            llm_model="gpt-3.5-turbo-16k"
        )
        
        # Convert the list of net types to a formatted string
        net_types_str = ", ".join([f"'{net}'" for net in net_types])
        
        # Run the extraction
        result = program(paper_text=paper_text, net_types=net_types_str)
        
        # Convert Pydantic model to list of dictionaries
        extracted_data = [net_item.dict() for net_item in result.net_data]
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error in LlamaIndex extraction: {str(e)}")
        return []

def extract_net_data(paper_text: str, schema: Dict[str, str], net_types: List[str], llm_service: str = "openai", api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Extract structured data from a research paper for multiple net types.
    
    Args:
        paper_text: The full text of the research paper
        schema: Dictionary defining the data schema
        net_types: List of net types to extract data for
        llm_service: The LLM service to use ('openai' or 'llamaindex')
        api_key: API key for the LLM service
        
    Returns:
        A pandas DataFrame containing rows for each net type
    """
    if not paper_text:
        logger.error("No paper text provided")
        return pd.DataFrame(columns=schema.keys())
    
    # Extract data using the specified LLM service
    if llm_service.lower() == "openai":
        extracted_data = extract_data_with_openai(paper_text, net_types, api_key)
    elif llm_service.lower() == "llamaindex":
        extracted_data = extract_data_with_llamaindex(paper_text, net_types, api_key)
    else:
        logger.error(f"Unsupported LLM service: {llm_service}")
        return pd.DataFrame(columns=schema.keys())
    
    # Handle extraction failure
    if not extracted_data:
        logger.error("Data extraction failed")
        return pd.DataFrame(columns=schema.keys())
    
    # Validate with Pydantic
    validated_data = []
    for item in extracted_data:
        try:
            # Add Net_type if missing
            if "Net_type" not in item and len(extracted_data) == 1 and len(net_types) == 1:
                item["Net_type"] = net_types[0]
                
            # Validate with Pydantic model
            net_data = NetData(**item)
            validated_data.append(net_data.dict())
        except Exception as e:
            logger.warning(f"Validation error for item {item}: {str(e)}")
            # Try to salvage what we can
            if "Net_type" in item:
                validated_data.append(item)
    
    # Convert to DataFrame
    df = pd.DataFrame(validated_data)
    
    # Check if all requested net types are in the results
    missing_nets = [net for net in net_types if net not in df["Net_type"].values]
    if missing_nets:
        logger.warning(f"Missing data for net types: {missing_nets}")
        
        # Add empty rows for missing net types
        for net in missing_nets:
            empty_row = {field: None for field in schema}
            empty_row["Net_type"] = net
            df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
    
    return df

# Example usage
if __name__ == "__main__":
    # Example schema matching the Pydantic model
    example_schema = {
        "Study_type": "str",
        "Net_type": "str",
        "Insecticide": "str",
        "Num_washes": "int",
        "Source": "str",
        "Mosquito_spp": "str",
        "Site": "str",
        "Start_date": "str",
        "End_date": "str",
        "Time_elapsed": "float"
    }
    
    # Example net types from the paper
    example_net_types = ["Interceptor G2", "Interceptor", "untreated net"]
    
    # For testing
    example_paper_text = """
    The full paper text would be pasted here.
    
    For demonstration purposes, this is a placeholder for the full text of the paper
    available at https://doi.org/10.1186/s12936-019-2973-x, which discusses
    the evaluation of Interceptor G2 LLIN compared to standard Interceptor LLIN
    and an untreated control net.
    """
    
    print("To extract data from a paper:")
    print("1. Import the function: from paper_extractor_v2 import extract_net_data")
    print("2. Call the function with your paper text, schema, net types, and API key:")
    print("   df = extract_net_data(paper_text, schema, net_types, api_key='your_api_key')")
    
    # To run with actual paper text, uncomment and populate:
    # Replace with your API key
    # api_key = "your_api_key"
    # df = extract_net_data(example_paper_text, example_schema, example_net_types, api_key=api_key)
    # print(df) 