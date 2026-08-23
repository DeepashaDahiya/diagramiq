"""
DiagramIQ-Eval Benchmark Dataset Generator
Generates reproducible PlantUML source specifications and ground-truth QA pairs
for the 100-diagram research benchmark.
"""

import os
import json
import csv
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
PUML_DIR = BENCHMARK_DIR / "plantuml_sources"
PUML_DIR.mkdir(parents=True, exist_ok=True)

# Sample PlantUML specifications representing core categories
SAMPLE_PUMLS = {
    "uml_class_ecommerce.puml": """@startuml
class User {
  +String userId
  +String email
  +login()
  +logout()
}

class Customer {
  +String shippingAddress
  +placeOrder()
}

class Order {
  +String orderId
  +Date orderDate
  +calculateTotal()
}

class Payment {
  +String paymentId
  +Float amount
  +processPayment()
}

User <|-- Customer
Customer "1" *-- "many" Order
Order "1" o-- "1" Payment
@enduml
""",
    "architecture_microservices.puml": """@startuml
node "Client Browser" as Client
node "API Gateway" as Gateway
node "Auth Service" as Auth
node "Order Service" as OrderSvc
node "Payment Service" as PaySvc
database "User DB" as UserDB
database "Order DB" as OrderDB

Client -> Gateway : HTTPS Request
Gateway -> Auth : Verify Token
Auth -> UserDB : Query Credentials
Gateway -> OrderSvc : Route Order
OrderSvc -> OrderDB : Persist Record
OrderSvc -> PaySvc : Initiate Payment
@enduml
""",
    "uml_sequence_checkout.puml": """@startuml
actor Customer
participant "Frontend UI" as UI
participant "Payment Gateway" as PG
database "Banking Network" as Bank

Customer -> UI : Click 'Pay Now'
activate UI
UI -> PG : Authorize Charge ($99)
activate PG
PG -> Bank : Submit Card Token
activate Bank
Bank --> PG : Approval Code
deactivate Bank
PG --> UI : 200 OK (Payment Success)
deactivate PG
UI --> Customer : Display Order Confirmation
deactivate UI
@enduml
"""
}

def generate_benchmark_assets():
    print("[*] Generating DiagramIQ-Eval PlantUML Sources & QA Annotations...")
    
    # 1. Write PUML files
    for filename, content in SAMPLE_PUMLS.items():
        puml_file = PUML_DIR / filename
        with open(puml_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"    [+] Created PlantUML source: {puml_file.name}")

    # 2. Generate Ground-Truth QA Annotation CSV
    csv_file = BENCHMARK_DIR / "diagramiq_eval_qa_benchmark.csv"
    qa_records = [
        {
            "diagram_id": "uml_class_ecommerce",
            "diagram_type": "UML_CLASS",
            "complexity": "MEDIUM",
            "question": "What is the relationship between Customer and User?",
            "reference_answer": "Customer inherits from User (generalization relationship), meaning Customer possesses all attributes and methods of User.",
            "question_type": "relationship_reasoning"
        },
        {
            "diagram_id": "uml_class_ecommerce",
            "diagram_type": "UML_CLASS",
            "complexity": "MEDIUM",
            "question": "What happens if the Order entity is removed?",
            "reference_answer": "Removing Order severs the connection between Customer and Payment, causing order placement and total calculation to break.",
            "question_type": "structural_analysis"
        },
        {
            "diagram_id": "architecture_microservices",
            "diagram_type": "SW_ARCHITECTURE",
            "complexity": "MEDIUM",
            "question": "Which component is a Single Point of Failure (SPOF)?",
            "reference_answer": "The API Gateway and Auth Service are Single Points of Failure because all downstream services (Order and Payment) depend on their operational status.",
            "question_type": "failure_impact"
        },
        {
            "diagram_id": "architecture_microservices",
            "diagram_type": "SW_ARCHITECTURE",
            "complexity": "MEDIUM",
            "question": "Explain the request flow from Client to Order DB.",
            "reference_answer": "Client requests flow through the API Gateway, which authenticates via Auth Service, forwards to Order Service, and writes to Order DB.",
            "question_type": "flow_tracing"
        },
        {
            "diagram_id": "uml_sequence_checkout",
            "diagram_type": "UML_SEQUENCE",
            "complexity": "MEDIUM",
            "question": "Which participant initiates the authorization request to the Banking Network?",
            "reference_answer": "The Payment Gateway participant receives the token from Frontend UI and submits the card token directly to the Banking Network.",
            "question_type": "component_identification"
        }
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["diagram_id", "diagram_type", "complexity", "question", "reference_answer", "question_type"])
        writer.writeheader()
        writer.writerows(qa_records)

    print(f"    [+] Saved QA benchmark annotations: {csv_file.name}")
    print("[SUCCESS] DiagramIQ-Eval benchmark assets created.")

if __name__ == "__main__":
    generate_benchmark_assets()
