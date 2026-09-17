"""
DiagramIQ-Eval Benchmark Dataset Generator
Generates the complete 100-diagram research benchmark:
  - 30 UML Class Diagrams (Simple, Medium, Complex)
  - 25 Microservice & System Architecture Diagrams
  - 25 UML Sequence Diagrams
  - 20 ER Diagrams and Control Flowcharts
Produces:
  1. Rendered high-resolution PNG diagram images in `data/diagramiq_eval/images/`
  2. PlantUML/Specification sources in `data/diagramiq_eval/plantuml_sources/`
  3. Ground-truth NetworkX graph JSONs in `data/diagramiq_eval/ground_truth_graphs/`
  4. Master 300-item QA benchmark CSV in `data/diagramiq_eval/diagramiq_eval_qa_benchmark.csv`
"""

import os
import sys
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BENCHMARK_DIR / "images"
PUML_DIR = BENCHMARK_DIR / "plantuml_sources"
GT_GRAPHS_DIR = BENCHMARK_DIR / "ground_truth_graphs"
CSV_FILE = BENCHMARK_DIR / "diagramiq_eval_qa_benchmark.csv"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PUML_DIR.mkdir(parents=True, exist_ok=True)
GT_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 100 BENCHMARK DIAGRAM SPECIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_all_benchmark_definitions():
    """Returns metadata, nodes, edges, PlantUML source, and 3 QA pairs for each of the 100 diagrams."""
    benchmarks = []

    # ── 1. 30 UML CLASS DIAGRAMS ──────────────────────────────────────────────
    uml_class_configs = [
        # (id, name, complexity, nodes, edges, puml, [qa1, qa2, qa3])
        (
            "uml_class_001", "E-Commerce User & Order System", "MEDIUM",
            [
                {"id": "User", "type": "class", "attrs": ["+userId", "+email"], "methods": ["+login()", "+logout()"]},
                {"id": "Customer", "type": "class", "attrs": ["+shippingAddress"], "methods": ["+placeOrder()"]},
                {"id": "Order", "type": "class", "attrs": ["+orderId", "+orderDate"], "methods": ["+calculateTotal()"]},
                {"id": "Payment", "type": "class", "attrs": ["+paymentId", "+amount"], "methods": ["+process()"]},
                {"id": "Product", "type": "class", "attrs": ["+sku", "+price"], "methods": ["+updateStock()"]}
            ],
            [
                {"from": "Customer", "to": "User", "type": "inherits"},
                {"from": "Customer", "to": "Order", "type": "places"},
                {"from": "Order", "to": "Payment", "type": "has_payment"},
                {"from": "Order", "to": "Product", "type": "contains_item"}
            ],
            [
                ("What is the relationship between Customer and User?", "Customer inherits from User (generalization/inheritance), sharing its identity and authentication attributes.", "relationship_reasoning"),
                ("What component connects Order to financial transactions?", "The Payment class processes financial settlements linked directly to the Order entity.", "component_identification"),
                ("If the Order class is removed, what breaks?", "The structural link between Customer and Payment/Product is severed, breaking order placement, totaling, and payments.", "failure_impact")
            ]
        ),
        (
            "uml_class_002", "Banking & Account Management", "MEDIUM",
            [
                {"id": "BankAccount", "type": "abstract_class", "attrs": ["#accountNumber", "#balance"], "methods": ["+deposit()", "+withdraw()*"]},
                {"id": "SavingsAccount", "type": "class", "attrs": ["+interestRate"], "methods": ["+applyInterest()", "+withdraw()"]},
                {"id": "CheckingAccount", "type": "class", "attrs": ["+overdraftLimit"], "methods": ["+withdraw()"]},
                {"id": "Transaction", "type": "class", "attrs": ["+txId", "+amount", "+timestamp"], "methods": ["+execute()"]},
                {"id": "BankCustomer", "type": "class", "attrs": ["+name", "+ssn"], "methods": ["+openAccount()"]}
            ],
            [
                {"from": "SavingsAccount", "to": "BankAccount", "type": "inherits"},
                {"from": "CheckingAccount", "to": "BankAccount", "type": "inherits"},
                {"from": "BankCustomer", "to": "BankAccount", "type": "owns"},
                {"from": "BankAccount", "to": "Transaction", "type": "records"}
            ],
            [
                ("Which classes inherit from BankAccount?", "SavingsAccount and CheckingAccount inherit from the abstract BankAccount base class.", "relationship_reasoning"),
                ("What represents the link between BankCustomer and accounts?", "The owns association links BankCustomer to one or more BankAccount instances.", "component_identification"),
                ("What happens if BankAccount is deleted?", "The inheritance hierarchy collapses, disabling both SavingsAccount and CheckingAccount.", "failure_impact")
            ]
        ),
        (
            "uml_class_003", "Hospital Management System", "COMPLEX",
            [
                {"id": "Person", "type": "class", "attrs": ["+id", "+name"], "methods": ["+getInfo()"]},
                {"id": "Doctor", "type": "class", "attrs": ["+specialization"], "methods": ["+prescribe()", "+diagnose()"]},
                {"id": "Patient", "type": "class", "attrs": ["+medicalHistory"], "methods": ["+admit()", "+discharge()"]},
                {"id": "Appointment", "type": "class", "attrs": ["+dateTime", "+status"], "methods": ["+schedule()", "+cancel()"]},
                {"id": "Prescription", "type": "class", "attrs": ["+dosage", "+medication"], "methods": ["+issue()"]},
                {"id": "Department", "type": "class", "attrs": ["+deptCode", "+name"], "methods": ["+assignDoctor()"]}
            ],
            [
                {"from": "Doctor", "to": "Person", "type": "inherits"},
                {"from": "Patient", "to": "Person", "type": "inherits"},
                {"from": "Doctor", "to": "Appointment", "type": "attends"},
                {"from": "Patient", "to": "Appointment", "type": "books"},
                {"from": "Doctor", "to": "Prescription", "type": "writes"},
                {"from": "Doctor", "to": "Department", "type": "belongs_to"}
            ],
            [
                ("Identify the shared base class for Doctor and Patient.", "Both Doctor and Patient inherit from the Person parent class.", "component_identification"),
                ("Which entity associates Doctor and Patient for a consultation?", "The Appointment class mediates the scheduling relationship between Doctor and Patient.", "relationship_reasoning"),
                ("What downstream entity is impacted if Doctor is removed?", "Prescriptions cannot be authored and Department doctor assignments break.", "failure_impact")
            ]
        ),
        (
            "uml_class_004", "University Course Registration", "COMPLEX",
            [
                {"id": "Student", "type": "class", "attrs": ["+studentId", "+major"], "methods": ["+registerCourse()"]},
                {"id": "Instructor", "type": "class", "attrs": ["+facultyId", "+rank"], "methods": ["+assignGrade()"]},
                {"id": "Course", "type": "class", "attrs": ["+courseCode", "+credits"], "methods": ["+getSyllabus()"]},
                {"id": "Section", "type": "class", "attrs": ["+sectionNum", "+room"], "methods": ["+getCapacity()"]},
                {"id": "Enrollment", "type": "class", "attrs": ["+grade", "+term"], "methods": ["+calcGPA()"]},
                {"id": "Department", "type": "class", "attrs": ["+name"], "methods": ["+offerCourse()"]}
            ],
            [
                {"from": "Student", "to": "Enrollment", "type": "has_enrollment"},
                {"from": "Enrollment", "to": "Section", "type": "enrolled_in"},
                {"from": "Section", "to": "Course", "type": "instance_of"},
                {"from": "Instructor", "to": "Section", "type": "teaches"},
                {"from": "Department", "to": "Course", "type": "offers"}
            ],
            [
                ("How does a Student connect to a Course?", "A Student connects via an Enrollment record that references a specific Course Section.", "relationship_reasoning"),
                ("Which class manages physical room and section scheduling?", "The Section class holds room location, section number, and capacity.", "component_identification"),
                ("What happens if Enrollment is removed?", "The association between Student academic records and Course Sections is broken.", "failure_impact")
            ]
        ),
        (
            "uml_class_005", "Ride-Sharing Driver & Rider Model", "MEDIUM",
            [
                {"id": "Rider", "type": "class", "attrs": ["+riderId", "+rating"], "methods": ["+requestRide()"]},
                {"id": "Driver", "type": "class", "attrs": ["+driverId", "+licenseNum"], "methods": ["+acceptRide()"]},
                {"id": "Ride", "type": "class", "attrs": ["+pickupLoc", "+dropoffLoc", "+fare"], "methods": ["+start()", "+complete()"]},
                {"id": "Vehicle", "type": "class", "attrs": ["+plate", "+model"], "methods": ["+inspect()"]},
                {"id": "TripPayment", "type": "class", "attrs": ["+amount", "+tip"], "methods": ["+chargeRider()"]}
            ],
            [
                {"from": "Rider", "to": "Ride", "type": "requests"},
                {"from": "Driver", "to": "Ride", "type": "fulfills"},
                {"from": "Driver", "to": "Vehicle", "type": "drives"},
                {"from": "Ride", "to": "TripPayment", "type": "generates"}
            ],
            [
                ("Which class coordinates between Rider and Driver?", "The Ride class encapsulates the active trip contract connecting Rider and Driver.", "relationship_reasoning"),
                ("What entity captures financial payment for a Ride?", "The TripPayment class records the charged fare and driver tip.", "component_identification"),
                ("If Ride is removed, what is the impact on TripPayment?", "TripPayment loses its source context and cannot compute or allocate trip fares.", "failure_impact")
            ]
        )
    ]

    # Dynamically expand UML Class diagrams to 30 items
    class_domains = [
        ("uml_class_006", "Library Management System", ["Book", "Author", "Member", "Loan", "Librarian"], [("Member", "Loan"), ("Loan", "Book"), ("Author", "Book"), ("Librarian", "Loan")]),
        ("uml_class_007", "Flight Reservation Engine", ["Flight", "Airline", "Passenger", "Ticket", "Airport"], [("Passenger", "Ticket"), ("Ticket", "Flight"), ("Airline", "Flight"), ("Flight", "Airport")]),
        ("uml_class_008", "Hotel Booking Platform", ["Guest", "Room", "Booking", "Invoice", "Hotel"], [("Guest", "Booking"), ("Booking", "Room"), ("Booking", "Invoice"), ("Hotel", "Room")]),
        ("uml_class_009", "Social Media Network", ["UserProfile", "Post", "Comment", "MediaAttachment", "Like"], [("UserProfile", "Post"), ("Post", "Comment"), ("Post", "MediaAttachment"), ("UserProfile", "Like")]),
        ("uml_class_010", "Online Food Delivery", ["Customer", "Restaurant", "MenuItem", "Order", "Courier"], [("Customer", "Order"), ("Order", "MenuItem"), ("Restaurant", "MenuItem"), ("Courier", "Order")]),
        ("uml_class_011", "Video Streaming Platform", ["User", "Video", "Playlist", "Subscription", "Channel"], [("User", "Subscription"), ("Subscription", "Channel"), ("Channel", "Video"), ("User", "Playlist")]),
        ("uml_class_012", "Vehicle Rental System", ["Customer", "RentalContract", "Car", "InsurancePolicy", "Location"], [("Customer", "RentalContract"), ("RentalContract", "Car"), ("RentalContract", "InsurancePolicy"), ("Car", "Location")]),
        ("uml_class_013", "Smart Home IoT Architecture", ["SmartHub", "Device", "Sensor", "Actuator", "AutomationRule"], [("SmartHub", "Device"), ("Device", "Sensor"), ("Device", "Actuator"), ("SmartHub", "AutomationRule")]),
        ("uml_class_014", "Warehouse Inventory Controller", ["Warehouse", "StockItem", "Supplier", "PurchaseOrder", "Bin"], [("Warehouse", "Bin"), ("Bin", "StockItem"), ("Supplier", "PurchaseOrder"), ("PurchaseOrder", "StockItem")]),
        ("uml_class_015", "Real Estate MLS Platform", ["Property", "ListingAgent", "Buyer", "Offer", "InspectionReport"], [("Buyer", "Offer"), ("Offer", "Property"), ("ListingAgent", "Property"), ("Property", "InspectionReport")]),
        ("uml_class_016", "Learning Management System", ["Course", "Assignment", "Submission", "Student", "GradeBook"], [("Student", "Submission"), ("Submission", "Assignment"), ("Course", "Assignment"), ("Course", "GradeBook")]),
        ("uml_class_017", "RPG Game Entity System", ["Entity", "Character", "Weapon", "Inventory", "Spell"], [("Character", "Entity"), ("Character", "Inventory"), ("Inventory", "Weapon"), ("Character", "Spell")]),
        ("uml_class_018", "Document Management Engine", ["Document", "Folder", "Permission", "UserGroup", "VersionHistory"], [("Folder", "Document"), ("Document", "VersionHistory"), ("Document", "Permission"), ("UserGroup", "Permission")]),
        ("uml_class_019", "HR & Recruitment Pipeline", ["Applicant", "JobPosting", "Resume", "Interview", "Recruiter"], [("Applicant", "Resume"), ("Applicant", "Interview"), ("JobPosting", "Applicant"), ("Recruiter", "Interview")]),
        ("uml_class_020", "Helpdesk Ticketing System", ["Ticket", "Customer", "SupportAgent", "SLAContract", "Resolution"], [("Customer", "Ticket"), ("SupportAgent", "Ticket"), ("Ticket", "Resolution"), ("Ticket", "SLAContract")]),
        ("uml_class_021", "Gym Fitness Membership", ["Member", "Trainer", "WorkoutPlan", "GymClass", "Schedule"], [("Member", "WorkoutPlan"), ("Trainer", "WorkoutPlan"), ("Trainer", "GymClass"), ("Member", "Schedule")]),
        ("uml_class_022", "Project Task Management", ["Workspace", "Project", "Task", "SubTask", "Assignee"], [("Workspace", "Project"), ("Project", "Task"), ("Task", "SubTask"), ("Task", "Assignee")]),
        ("uml_class_023", "Music Streaming Catalog", ["Artist", "Album", "Track", "Playlist", "Listener"], [("Artist", "Album"), ("Album", "Track"), ("Listener", "Playlist"), ("Playlist", "Track")]),
        ("uml_class_024", "Parking Lot Controller", ["ParkingLot", "ParkingSpot", "ParkingTicket", "Vehicle", "PaymentKiosk"], [("ParkingLot", "ParkingSpot"), ("Vehicle", "ParkingTicket"), ("ParkingTicket", "ParkingSpot"), ("ParkingTicket", "PaymentKiosk")]),
        ("uml_class_025", "Telehealth Video Consultation", ["Doctor", "Patient", "ConsultationSession", "Prescription", "EHRRecord"], [("Doctor", "ConsultationSession"), ("Patient", "ConsultationSession"), ("ConsultationSession", "Prescription"), ("Patient", "EHRRecord")]),
        ("uml_class_026", "Online Auction Marketplace", ["Seller", "AuctionItem", "Bid", "Buyer", "EscrowAccount"], [("Seller", "AuctionItem"), ("Buyer", "Bid"), ("Bid", "AuctionItem"), ("AuctionItem", "EscrowAccount")]),
        ("uml_class_027", "Digital Crypto Wallet", ["Wallet", "PrivateKey", "TransactionRecord", "CryptoToken", "ExchangeNode"], [("Wallet", "PrivateKey"), ("Wallet", "TransactionRecord"), ("TransactionRecord", "CryptoToken"), ("Wallet", "ExchangeNode")]),
        ("uml_class_028", "Cloud VPC Infrastructure", ["VPC", "Subnet", "EC2Instance", "SecurityGroup", "RouteTable"], [("VPC", "Subnet"), ("Subnet", "EC2Instance"), ("EC2Instance", "SecurityGroup"), ("Subnet", "RouteTable")]),
        ("uml_class_029", "Supply Chain Logistics", ["Shipment", "Carrier", "RouteCheckpoint", "Consignee", "Manifest"], [("Carrier", "Shipment"), ("Shipment", "RouteCheckpoint"), ("Shipment", "Consignee"), ("Shipment", "Manifest")]),
        ("uml_class_030", "CDN Edge Cache Model", ["EdgeServer", "OriginServer", "CacheEntry", "RoutingRule", "ClientSession"], [("EdgeServer", "CacheEntry"), ("EdgeServer", "OriginServer"), ("ClientSession", "EdgeServer"), ("EdgeServer", "RoutingRule")])
    ]

    for cid, name, nodes, edges in class_domains:
        node_objs = [{"id": n, "type": "class", "attrs": [f"+{n.lower()}Id"], "methods": ["+execute()"]} for n in nodes]
        edge_objs = [{"from": u, "to": v, "type": "associates"} for u, v in edges]
        qa = [
            (f"What is the primary role of {nodes[0]} in {name}?", f"{nodes[0]} acts as the primary root controller or initiator in the {name}.", "component_identification"),
            (f"How does {nodes[0]} relate to {nodes[1]}?", f"{nodes[0]} connects to {nodes[1]} to coordinate core domain functionality.", "relationship_reasoning"),
            (f"What happens if {nodes[1]} is disconnected?", f"Removing {nodes[1]} breaks dependent associations and downstream transaction flows in {name}.", "failure_impact")
        ]
        uml_class_configs.append((cid, name, "MEDIUM", node_objs, edge_objs, qa))

    # Add all 30 UML class benchmarks
    for cid, name, comp, nodes, edges, qa_tuples in uml_class_configs:
        benchmarks.append({
            "diagram_id": cid,
            "title": name,
            "category": "UML_CLASS",
            "complexity": comp,
            "nodes": nodes,
            "edges": edges,
            "qa_pairs": [
                {"question": q, "reference_answer": a, "question_type": t} for q, a, t in qa_tuples
            ]
        })

    # ── 2. 25 MICROSERVICE & ARCHITECTURE DIAGRAMS ────────────────────────────
    arch_configs = [
        (
            "arch_microservice_001", "E-Commerce Microservices Architecture", "MEDIUM",
            ["ClientApp", "APIGateway", "AuthService", "OrderService", "PaymentGateway", "UserDB", "OrderDB"],
            [("ClientApp", "APIGateway"), ("APIGateway", "AuthService"), ("AuthService", "UserDB"), ("APIGateway", "OrderService"), ("OrderService", "OrderDB"), ("OrderService", "PaymentGateway")],
            [
                ("Identify the Single Point of Failure (SPOF) in this architecture.", "The APIGateway is a Single Point of Failure (SPOF) because all client requests must traverse it to reach downstream services.", "failure_impact"),
                ("Explain the end-to-end data flow from ClientApp to OrderDB.", "ClientApp sends HTTPS requests to APIGateway, which authenticates via AuthService and forwards orders to OrderService, writing to OrderDB.", "flow_tracing"),
                ("What does OrderService depend on?", "OrderService depends upstream on APIGateway and downstream on OrderDB and PaymentGateway.", "relationship_reasoning")
            ]
        ),
        (
            "arch_microservice_002", "Netflix-style Video Streaming Cloud", "COMPLEX",
            ["UserDevice", "CloudFrontCDN", "APIGateway", "TranscodingWorker", "S3Storage", "MetadataDB", "AuthServer"],
            [("UserDevice", "CloudFrontCDN"), ("UserDevice", "APIGateway"), ("APIGateway", "AuthServer"), ("APIGateway", "TranscodingWorker"), ("TranscodingWorker", "S3Storage"), ("TranscodingWorker", "MetadataDB"), ("CloudFrontCDN", "S3Storage")],
            [
                ("How are video assets served to the UserDevice?", "Static video chunks are cached and served via CloudFrontCDN directly from S3Storage.", "flow_tracing"),
                ("What role does TranscodingWorker play?", "TranscodingWorker processes incoming video, stores transcode renditions in S3Storage, and updates MetadataDB.", "component_identification"),
                ("What happens if AuthServer goes down?", "New user sessions cannot authenticate through APIGateway, preventing API access while cached CDN streams remain active.", "failure_impact")
            ]
        ),
        (
            "arch_microservice_003", "Real-Time Geo-Location Tracking System", "COMPLEX",
            ["MobileClient", "WebSocketGateway", "LocationService", "RedisGeoCache", "KafkaStream", "DispatchEngine", "PostgresDB"],
            [("MobileClient", "WebSocketGateway"), ("WebSocketGateway", "LocationService"), ("LocationService", "RedisGeoCache"), ("LocationService", "KafkaStream"), ("KafkaStream", "DispatchEngine"), ("DispatchEngine", "PostgresDB")],
            [
                ("Which component handles high-throughput asynchronous location event queuing?", "KafkaStream acts as the distributed event buffer between LocationService and DispatchEngine.", "component_identification"),
                ("Why is RedisGeoCache placed before PostgresDB?", "RedisGeoCache provides sub-millisecond in-memory spatial indexing for real-time driver proximity queries.", "structural_analysis"),
                ("Trace the blast radius if LocationService crashes.", "Downstream components (RedisGeoCache, KafkaStream, and DispatchEngine) stop receiving live telemetry updates.", "failure_impact")
            ]
        ),
        (
            "arch_microservice_004", "Real-time Chat Messaging Architecture", "MEDIUM",
            ["WebClient", "LoadBalancer", "ChatServer", "RedisPubSub", "CassandraDB"],
            [("WebClient", "LoadBalancer"), ("LoadBalancer", "ChatServer"), ("ChatServer", "RedisPubSub"), ("ChatServer", "CassandraDB")],
            [
                ("What is the role of RedisPubSub in this chat system?", "RedisPubSub broadcasts real-time chat messages across distributed ChatServer instances.", "component_identification"),
                ("Which component is the database of record for chat history?", "CassandraDB provides partitioned persistence for message history and chat logs.", "relationship_reasoning"),
                ("What happens if LoadBalancer fails?", "WebClient traffic cannot be routed to ChatServer instances, causing a total chat outage.", "failure_impact")
            ]
        ),
        (
            "arch_microservice_005", "Fintech Payment Switch & Ledger", "COMPLEX",
            ["POSMerchant", "APIGateway", "FraudDetection", "CardTokenizer", "BankSwitch", "LedgerDB"],
            [("POSMerchant", "APIGateway"), ("APIGateway", "FraudDetection"), ("APIGateway", "CardTokenizer"), ("CardTokenizer", "BankSwitch"), ("BankSwitch", "LedgerDB")],
            [
                ("Which service executes fraud risk evaluation before payment authorization?", "FraudDetection inspects transaction metadata directly behind APIGateway.", "component_identification"),
                ("Explain the flow between CardTokenizer and BankSwitch.", "CardTokenizer converts sensitive PAN into a secure token and forwards it to BankSwitch for banking settlement.", "flow_tracing"),
                ("What breaks if LedgerDB is unreachable?", "BankSwitch transactions cannot be reconciled or permanently audited, blocking final settlement.", "failure_impact")
            ]
        )
    ]

    # Expand to 25 Architecture benchmarks
    arch_domains = [
        ("arch_microservice_006", "Distributed Search Ingestion Pipeline", ["WebCrawler", "KafkaQueue", "SparkParser", "Elasticsearch", "SearchAPI"], [("WebCrawler", "KafkaQueue"), ("KafkaQueue", "SparkParser"), ("SparkParser", "Elasticsearch"), ("SearchAPI", "Elasticsearch")]),
        ("arch_microservice_007", "IoT Sensor Ingestion & Telemetry Lake", ["IoTSensors", "MQTTBroker", "FlinkEngine", "TimescaleDB", "GrafanaDashboard"], [("IoTSensors", "MQTTBroker"), ("MQTTBroker", "FlinkEngine"), ("FlinkEngine", "TimescaleDB"), ("TimescaleDB", "GrafanaDashboard")]),
        ("arch_microservice_008", "Cloud Single Sign-On Identity System", ["BrowserClient", "OAuthGateway", "KeycloakIDP", "LDAPDirectory", "SessionRedis"], [("BrowserClient", "OAuthGateway"), ("OAuthGateway", "KeycloakIDP"), ("KeycloakIDP", "LDAPDirectory"), ("KeycloakIDP", "SessionRedis")]),
        ("arch_microservice_009", "Multi-Channel Notification Hub", ["EventProducer", "SQSQueue", "NotificationWorker", "TwilioSMS", "SendGridEmail"], [("EventProducer", "SQSQueue"), ("SQSQueue", "NotificationWorker"), ("NotificationWorker", "TwilioSMS"), ("NotificationWorker", "SendGridEmail")]),
        ("arch_microservice_010", "Medallion Data Lakehouse Architecture", ["SourceDB", "BronzeLayer", "SparkETL", "SilverLayer", "GoldAnalytics"], [("SourceDB", "BronzeLayer"), ("BronzeLayer", "SparkETL"), ("SparkETL", "SilverLayer"), ("SilverLayer", "GoldAnalytics")]),
        ("arch_microservice_011", "AI Recommendation Vector Search", ["UserEvents", "EventCollector", "EmbeddingModel", "MilvusVectorDB", "RankerService"], [("UserEvents", "EventCollector"), ("EventCollector", "EmbeddingModel"), ("EmbeddingModel", "MilvusVectorDB"), ("RankerService", "MilvusVectorDB")]),
        ("arch_microservice_012", "Serverless Event-Driven Order Processing", ["APIGW", "LambdaAuth", "LambdaOrder", "DynamoDB", "EventBridge"], [("APIGW", "LambdaAuth"), ("APIGW", "LambdaOrder"), ("LambdaOrder", "DynamoDB"), ("LambdaOrder", "EventBridge")]),
        ("arch_microservice_013", "Multi-Region Cloud Disaster Recovery", ["Route53DNS", "RegionAGateway", "RegionAPostgres", "ReplicationPipe", "RegionBPostgres"], [("Route53DNS", "RegionAGateway"), ("RegionAGateway", "RegionAPostgres"), ("RegionAPostgres", "ReplicationPipe"), ("ReplicationPipe", "RegionBPostgres")]),
        ("arch_microservice_014", "Zero-Trust Mesh with OPA Policy", ["ServiceA", "EnvoyProxyA", "OPAPolicyEngine", "EnvoyProxyB", "ServiceB"], [("ServiceA", "EnvoyProxyA"), ("EnvoyProxyA", "OPAPolicyEngine"), ("EnvoyProxyA", "EnvoyProxyB"), ("EnvoyProxyB", "ServiceB")]),
        ("arch_microservice_015", "Kubernetes Ingress & Pod Fleet", ["IngressController", "ClusterIPService", "PodReplicas", "PostgresStatefulSet", "PVStorage"], [("IngressController", "ClusterIPService"), ("ClusterIPService", "PodReplicas"), ("PodReplicas", "PostgresStatefulSet"), ("PostgresStatefulSet", "PVStorage")]),
        ("arch_microservice_016", "High-Frequency Trading Engine", ["MarketDataFeed", "FIXGateway", "OrderMatcher", "RiskEngine", "FastJournal"], [("MarketDataFeed", "FIXGateway"), ("FIXGateway", "RiskEngine"), ("RiskEngine", "OrderMatcher"), ("OrderMatcher", "FastJournal")]),
        ("arch_microservice_017", "Dynamic Ride Pricing & Surge Engine", ["TripDemandTracker", "WeatherAPI", "SurgeCalculator", "PricingCache", "DispatchAPI"], [("TripDemandTracker", "SurgeCalculator"), ("WeatherAPI", "SurgeCalculator"), ("SurgeCalculator", "PricingCache"), ("PricingCache", "DispatchAPI")]),
        ("arch_microservice_018", "Automated Content Moderation Pipeline", ["UploadService", "KafkaBuffer", "VisionAIWorker", "TextNLPWorker", "ModerationDB"], [("UploadService", "KafkaBuffer"), ("KafkaBuffer", "VisionAIWorker"), ("KafkaBuffer", "TextNLPWorker"), ("VisionAIWorker", "ModerationDB")]),
        ("arch_microservice_019", "Healthcare FHIR Interoperability Bridge", ["HospitalEHR", "MLLPAdapter", "FHIRTransformer", "SecureVaultDB", "ClinicalPortal"], [("HospitalEHR", "MLLPAdapter"), ("MLLPAdapter", "FHIRTransformer"), ("FHIRTransformer", "SecureVaultDB"), ("ClinicalPortal", "SecureVaultDB")]),
        ("arch_microservice_020", "Distributed Cache-Aside Cluster", ["WebServer", "RedisCluster", "CacheMissHandler", "MasterDB", "ReplicaDB"], [("WebServer", "RedisCluster"), ("WebServer", "CacheMissHandler"), ("CacheMissHandler", "MasterDB"), ("MasterDB", "ReplicaDB")]),
        ("arch_microservice_021", "Ad-Tech Real-Time Bidding Exchange", ["AdPublisher", "ExchangeGateway", "BIDEvaluator", "BudgetPacer", "AdStorage"], [("AdPublisher", "ExchangeGateway"), ("ExchangeGateway", "BIDEvaluator"), ("BIDEvaluator", "BudgetPacer"), ("ExchangeGateway", "AdStorage")]),
        ("arch_microservice_022", "Mobile Banking Microservices Gateway", ["MobileApp", "WAFSecurity", "APIGateway", "CoreBankingHost", "SMS2FAService"], [("MobileApp", "WAFSecurity"), ("WAFSecurity", "APIGateway"), ("APIGateway", "CoreBankingHost"), ("APIGateway", "SMS2FAService")]),
        ("arch_microservice_023", "Automated CI/CD Build Cloud", ["GitWebhook", "BuildOrchestrator", "DockerWorkerPool", "ArtifactRegistry", "KubernetesCluster"], [("GitWebhook", "BuildOrchestrator"), ("BuildOrchestrator", "DockerWorkerPool"), ("DockerWorkerPool", "ArtifactRegistry"), ("ArtifactRegistry", "KubernetesCluster")]),
        ("arch_microservice_024", "Global CDN Edge Compute Mesh", ["EndUser", "AnycastDNS", "CloudflareEdge", "OriginBackend", "KVStorage"], [("EndUser", "AnycastDNS"), ("AnycastDNS", "CloudflareEdge"), ("CloudflareEdge", "KVStorage"), ("CloudflareEdge", "OriginBackend")]),
        ("arch_microservice_025", "Enterprise Data Governance Catalog", ["SchemaCrawler", "DataLineageEngine", "CatalogDB", "PrivacyScanner", "DataStewardUI"], [("SchemaCrawler", "DataLineageEngine"), ("DataLineageEngine", "CatalogDB"), ("PrivacyScanner", "CatalogDB"), ("DataStewardUI", "CatalogDB")])
    ]

    for aid, name, nodes, edges in arch_domains:
        qa = [
            (f"What is the function of {nodes[0]} in {name}?", f"{nodes[0]} serves as the entry ingestion node for incoming traffic in {name}.", "component_identification"),
            (f"Trace the primary data path from {nodes[0]} to {nodes[-1]}.", f"Requests flow from {nodes[0]} through intermediary processors to terminate at {nodes[-1]}.", "flow_tracing"),
            (f"Which node represents a single bottleneck if {nodes[1]} fails?", f"If {nodes[1]} fails, traffic cannot reach downstream services, causing operational stoppage.", "failure_impact")
        ]
        arch_configs.append((aid, name, "MEDIUM", nodes, edges, qa))

    for aid, name, comp, nodes, edges, qa_tuples in arch_configs:
        node_objs = [{"id": n, "type": "service" if "Service" in n or "Worker" in n else ("database" if "DB" in n or "Cache" in n or "Storage" in n else "gateway")} for n in nodes]
        edge_objs = [{"from": u, "to": v, "type": "calls" if "DB" not in v else "writes_to"} for u, v in edges]
        benchmarks.append({
            "diagram_id": aid,
            "title": name,
            "category": "SW_ARCHITECTURE",
            "complexity": comp,
            "nodes": node_objs,
            "edges": edge_objs,
            "qa_pairs": [
                {"question": q, "reference_answer": a, "question_type": t} for q, a, t in qa_tuples
            ]
        })

    # ── 3. 25 UML SEQUENCE DIAGRAMS ───────────────────────────────────────────
    seq_configs = [
        (
            "uml_sequence_001", "OAuth2 Authorization Code Grant Flow", "MEDIUM",
            ["User", "Browser", "ClientApp", "AuthServer", "ResourceServer"],
            [("User", "Browser"), ("Browser", "AuthServer"), ("AuthServer", "Browser"), ("Browser", "ClientApp"), ("ClientApp", "AuthServer"), ("ClientApp", "ResourceServer")],
            [
                ("Which entity issues the authorization code to the Browser?", "The AuthServer authenticates the user and returns the authorization code to the Browser.", "component_identification"),
                ("What does ClientApp exchange with AuthServer to get an access token?", "ClientApp posts the authorization code and client secret to AuthServer in exchange for an access token.", "relationship_reasoning"),
                ("Trace the final step where protected user data is retrieved.", "ClientApp presents the Bearer access token to the ResourceServer to fetch protected resources.", "flow_tracing")
            ]
        ),
        (
            "uml_sequence_002", "Stripe Credit Card Payment Flow", "MEDIUM",
            ["Customer", "FrontendUI", "BackendServer", "StripeAPI", "CardNetwork"],
            [("Customer", "FrontendUI"), ("FrontendUI", "BackendServer"), ("BackendServer", "StripeAPI"), ("StripeAPI", "CardNetwork"), ("CardNetwork", "StripeAPI"), ("StripeAPI", "BackendServer")],
            [
                ("Who initiates the charge authorization request?", "The BackendServer requests payment intent creation from the StripeAPI.", "component_identification"),
                ("What entity communicates directly with the CardNetwork?", "StripeAPI securely submits tokenized payment payloads to CardNetwork.", "relationship_reasoning"),
                ("What happens if StripeAPI returns a card declined error?", "BackendServer receives the rejection and prompts FrontendUI to display payment failure.", "failure_impact")
            ]
        ),
        (
            "uml_sequence_003", "Two-Factor Authentication SMS Verification", "MEDIUM",
            ["User", "WebPortal", "AuthEngine", "TwilioSMSGateway", "UserMobile"],
            [("User", "WebPortal"), ("WebPortal", "AuthEngine"), ("AuthEngine", "TwilioSMSGateway"), ("TwilioSMSGateway", "UserMobile"), ("User", "WebPortal"), ("WebPortal", "AuthEngine")],
            [
                ("Which service dispatches the OTP code over cellular network?", "TwilioSMSGateway dispatches the one-time passcode directly to UserMobile.", "component_identification"),
                ("Where is the user's entered OTP verified?", "AuthEngine compares the submitted OTP against the cryptographically hashed active session token.", "relationship_reasoning"),
                ("If TwilioSMSGateway fails, what is the user impact?", "The user never receives the SMS code and cannot complete the two-factor authentication handshake.", "failure_impact")
            ]
        ),
        (
            "uml_sequence_004", "Database Connection Pool Allocation", "SIMPLE",
            ["AppThread", "PoolManager", "JDBCDriver", "DatabaseServer"],
            [("AppThread", "PoolManager"), ("PoolManager", "JDBCDriver"), ("JDBCDriver", "DatabaseServer"), ("PoolManager", "AppThread")],
            [
                ("Which component intercepts connection requests from AppThread?", "PoolManager manages available pooled connections before delegating to JDBCDriver.", "component_identification"),
                ("What happens if all connections in PoolManager are leased?", "AppThread is blocked or queued until an existing connection is returned to the pool.", "failure_impact"),
                ("Trace the release step when a query completes.", "AppThread returns the connection handle to PoolManager to mark it reusable.", "flow_tracing")
            ]
        ),
        (
            "uml_sequence_005", "JWT Token Refresh Handshake", "MEDIUM",
            ["ClientSPA", "APIGateway", "AuthService", "RedisTokenBlacklist"],
            [("ClientSPA", "APIGateway"), ("APIGateway", "AuthService"), ("AuthService", "RedisTokenBlacklist"), ("AuthService", "ClientSPA")],
            [
                ("Why does AuthService query RedisTokenBlacklist during refresh?", "AuthService ensures the presented refresh token has not been revoked or blacklisted.", "relationship_reasoning"),
                ("What is returned to ClientSPA upon successful token refresh?", "A new signed short-lived JWT Access Token and rotated Refresh Token.", "component_identification"),
                ("What happens if RedisTokenBlacklist is unreachable?", "AuthService triggers defensive fallback or rejects the refresh token for security.", "failure_impact")
            ]
        )
    ]

    # Expand to 25 Sequence diagrams
    seq_domains = [
        ("uml_sequence_006", "Order Cancellation & Auto-Refund Flow", ["Buyer", "OrderService", "PaymentGateway", "InventoryService"], [("Buyer", "OrderService"), ("OrderService", "PaymentGateway"), ("OrderService", "InventoryService")]),
        ("uml_sequence_007", "WebSocket Connection Upgrade Handshake", ["Browser", "ReverseProxy", "WebSocketServer", "PubSubChannel"], [("Browser", "ReverseProxy"), ("ReverseProxy", "WebSocketServer"), ("WebSocketServer", "PubSubChannel")]),
        ("uml_sequence_008", "Password Reset via Secure Email Token", ["User", "WebPortal", "TokenService", "SMTPMailer"], [("User", "WebPortal"), ("WebPortal", "TokenService"), ("TokenService", "SMTPMailer")]),
        ("uml_sequence_009", "Direct S3 Pre-Signed File Upload", ["ClientApp", "BackendAPI", "S3Storage", "AntivirusWorker"], [("ClientApp", "BackendAPI"), ("BackendAPI", "S3Storage"), ("ClientApp", "S3Storage"), ("S3Storage", "AntivirusWorker")]),
        ("uml_sequence_010", "SAML 2.0 Enterprise Single Sign-On", ["Employee", "ServiceProvider", "SAMLIdentityServer", "LDAPServer"], [("Employee", "ServiceProvider"), ("ServiceProvider", "SAMLIdentityServer"), ("SAMLIdentityServer", "LDAPServer")]),
        ("uml_sequence_011", "Circuit Breaker Tripping on Service Failure", ["ClientCaller", "APIGateway", "FailingService", "FallbackCache"], [("ClientCaller", "APIGateway"), ("APIGateway", "FailingService"), ("APIGateway", "FallbackCache")]),
        ("uml_sequence_012", "Cache-Aside Read Flow with Cache Miss", ["Application", "RedisCache", "PostgresDB", "MonitoringLogger"], [("Application", "RedisCache"), ("Application", "PostgresDB"), ("Application", "RedisCache")]),
        ("uml_sequence_013", "User Email Account Verification", ["NewUser", "FrontendApp", "RegistrationAPI", "EmailDispatcher"], [("NewUser", "FrontendApp"), ("FrontendApp", "RegistrationAPI"), ("RegistrationAPI", "EmailDispatcher")]),
        ("uml_sequence_014", "Ride Allocation & Driver Acceptance", ["Rider", "DispatchEngine", "DriverMobile", "TripTracker"], [("Rider", "DispatchEngine"), ("DispatchEngine", "DriverMobile"), ("DriverMobile", "DispatchEngine")]),
        ("uml_sequence_015", "Distributed Two-Phase Commit Protocol", ["TxCoordinator", "ParticipantA", "ParticipantB", "AuditLog"], [("TxCoordinator", "ParticipantA"), ("TxCoordinator", "ParticipantB"), ("TxCoordinator", "AuditLog")]),
        ("uml_sequence_016", "Push Notification Broadcast via FCM", ["BackendService", "FCMGateway", "APNsGateway", "MobilePhone"], [("BackendService", "FCMGateway"), ("FCMGateway", "MobilePhone")]),
        ("uml_sequence_017", "Adaptive Video Bitrate Chunk Request", ["VideoPlayer", "ManifestServer", "EdgeCDNServer", "DRMLicenseServer"], [("VideoPlayer", "ManifestServer"), ("VideoPlayer", "DRMLicenseServer"), ("VideoPlayer", "EdgeCDNServer")]),
        ("uml_sequence_018", "Real-Time Credit Card Fraud Scoring", ["POSTerminal", "BankAcquirer", "FraudScoringEngine", "CardNetwork"], [("POSTerminal", "BankAcquirer"), ("BankAcquirer", "FraudScoringEngine"), ("BankAcquirer", "CardNetwork")]),
        ("uml_sequence_019", "Live Auction Real-Time Bid Broadcast", ["Bidder", "AuctionWSServer", "BidValidator", "BidBroadcaster"], [("Bidder", "AuctionWSServer"), ("AuctionWSServer", "BidValidator"), ("AuctionWSServer", "BidBroadcaster")]),
        ("uml_sequence_020", "Git Push SSH Key Exchange & Webhook", ["DeveloperCLI", "SSHDaemon", "GitRepository", "CIWebhookTrigger"], [("DeveloperCLI", "SSHDaemon"), ("SSHDaemon", "GitRepository"), ("GitRepository", "CIWebhookTrigger")]),
        ("uml_sequence_021", "SSL/TLS 1.3 Cryptographic Handshake", ["WebClient", "WebServer", "CertificateAuthority"], [("WebClient", "WebServer"), ("WebServer", "WebClient")]),
        ("uml_sequence_022", "Distributed Redlock Mutex Acquisition", ["WorkerNode", "RedisMaster1", "RedisMaster2", "RedisMaster3"], [("WorkerNode", "RedisMaster1"), ("WorkerNode", "RedisMaster2"), ("WorkerNode", "RedisMaster3")]),
        ("uml_sequence_023", "Webhook Delivery with Retry Backoff", ["PublisherApp", "WebhookWorker", "SubscriberEndpoint", "DeadLetterQueue"], [("PublisherApp", "WebhookWorker"), ("WebhookWorker", "SubscriberEndpoint"), ("WebhookWorker", "DeadLetterQueue")]),
        ("uml_sequence_024", "Hotel Room Hold & Booking Settlement", ["Traveler", "BookingEngine", "InventoryDB", "PaymentProcessor"], [("Traveler", "BookingEngine"), ("BookingEngine", "InventoryDB"), ("BookingEngine", "PaymentProcessor")]),
        ("uml_sequence_025", "GitHub Actions CI/CD Deployment Flow", ["GitPushEvent", "GitHubRunner", "DockerRegistry", "KubernetesCluster"], [("GitPushEvent", "GitHubRunner"), ("GitHubRunner", "DockerRegistry"), ("GitHubRunner", "KubernetesCluster")])
    ]

    for sid, name, nodes, edges in seq_domains:
        qa = [
            (f"Which participant initiates message exchange in {name}?", f"{nodes[0]} initiates the chronological interaction in {name}.", "component_identification"),
            (f"Trace the sequential progression between {nodes[0]} and {nodes[1]}.", f"{nodes[0]} issues a request or synchronous message call to {nodes[1]}.", "flow_tracing"),
            (f"What happens if {nodes[-1]} fails to return an acknowledgment?", f"The sequence halts or triggers an error handler because {nodes[-1]} failed to acknowledge.", "failure_impact")
        ]
        seq_configs.append((sid, name, "MEDIUM", nodes, edges, qa))

    for sid, name, comp, nodes, edges, qa_tuples in seq_configs:
        node_objs = [{"id": n, "type": "actor" if idx == 0 else "participant"} for idx, n in enumerate(nodes)]
        edge_objs = [{"from": u, "to": v, "type": "sends_message"} for u, v in edges]
        benchmarks.append({
            "diagram_id": sid,
            "title": name,
            "category": "UML_SEQUENCE",
            "complexity": comp,
            "nodes": node_objs,
            "edges": edge_objs,
            "qa_pairs": [
                {"question": q, "reference_answer": a, "question_type": t} for q, a, t in qa_tuples
            ]
        })

    # ── 4. 20 ER DIAGRAMS & FLOWCHARTS ────────────────────────────────────────
    er_flowchart_configs = [
        # ER Diagrams (1-10)
        (
            "flowchart_er_001", "Hospital Patient & Department ERD", "ER_DIAGRAM", "MEDIUM",
            ["Patient", "Admission", "Doctor", "Department", "Prescription", "Room"],
            [("Patient", "Admission"), ("Admission", "Doctor"), ("Doctor", "Department"), ("Doctor", "Prescription"), ("Admission", "Room")],
            [
                ("Identify the foreign key relationship between Admission and Patient.", "Admission contains patient_id as a foreign key referencing the primary key of Patient.", "relationship_reasoning"),
                ("What cardinality exists between Doctor and Department?", "Many doctors belong to one Department (N:1 relationship).", "structural_analysis"),
                ("What happens if Patient records are purged?", "Dependent Admission and Prescription records will cascade delete or require anonymization.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_002", "University Academic Schema ERD", "ER_DIAGRAM", "MEDIUM",
            ["Student", "Enrollment", "Course", "Instructor", "Department"],
            [("Student", "Enrollment"), ("Enrollment", "Course"), ("Course", "Department"), ("Instructor", "Department")],
            [
                ("How are Student and Course linked in this relational model?", "Through the associative junction entity Enrollment, resolving a Many-to-Many (M:N) relationship.", "relationship_reasoning"),
                ("Which entity owns the course catalog?", "Department holds the 1:N relationship offering multiple Courses.", "component_identification"),
                ("What happens if Course is deleted?", "Enrollment records tied to that course become orphaned unless cascading deletes are enforced.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_003", "E-Commerce Product & Order Schema ERD", "ER_DIAGRAM", "MEDIUM",
            ["Customer", "OrderHeader", "OrderItem", "ProductCatalog", "PaymentRecord"],
            [("Customer", "OrderHeader"), ("OrderHeader", "OrderItem"), ("OrderItem", "ProductCatalog"), ("OrderHeader", "PaymentRecord")],
            [
                ("What is the role of OrderItem?", "OrderItem acts as a line-item table linking OrderHeader to specific ProductCatalog entries with quantities.", "component_identification"),
                ("Explain the relationship between Customer and OrderHeader.", "One Customer can place many OrderHeader records (1:N cardinality).", "relationship_reasoning"),
                ("If OrderHeader is deleted, what happens to PaymentRecord?", "PaymentRecord references order_id and loses its parent invoice association.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_004", "Hotel Reservation & Room ERD", "ER_DIAGRAM", "SIMPLE",
            ["Guest", "Reservation", "Room", "RoomType"],
            [("Guest", "Reservation"), ("Reservation", "Room"), ("Room", "RoomType")],
            [
                ("Identify the entity defining room rates and amenities.", "RoomType specifies category pricing and amenities referenced by individual Room instances.", "component_identification"),
                ("What cardinality links Guest to Reservation?", "One Guest can book multiple Reservation instances (1:N).", "relationship_reasoning"),
                ("What breaks if Room is deleted?", "Active Reservation records assigned to that room lose their physical allocation.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_005", "Airline Booking & Flight ERD", "ER_DIAGRAM", "MEDIUM",
            ["Passenger", "Booking", "FlightSchedule", "Aircraft", "Airport"],
            [("Passenger", "Booking"), ("Booking", "FlightSchedule"), ("FlightSchedule", "Aircraft"), ("FlightSchedule", "Airport")],
            [
                ("Which table links Passengers to specific Flights?", "The Booking entity stores ticket references linking Passenger to FlightSchedule.", "relationship_reasoning"),
                ("What does Aircraft associate with?", "Aircraft assigns a physical tail number and seat capacity to a FlightSchedule.", "component_identification"),
                ("What happens if FlightSchedule is canceled?", "All associated Booking instances must be refunded or rescheduled.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_006", "Real Estate Property Schema ERD", "ER_DIAGRAM", "SIMPLE",
            ["PropertyListing", "Agent", "Owner", "OfferRecord"],
            [("Owner", "PropertyListing"), ("Agent", "PropertyListing"), ("PropertyListing", "OfferRecord")],
            [
                ("Who lists the PropertyListing?", "Agent represents and lists PropertyListing on behalf of Owner.", "relationship_reasoning"),
                ("What captures buyer bids?", "OfferRecord stores submitted purchase offers for a specific PropertyListing.", "component_identification"),
                ("If PropertyListing is removed, what is the consequence?", "All associated OfferRecord entries become invalid.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_007", "Fleet Maintenance Log ERD", "ER_DIAGRAM", "SIMPLE",
            ["Vehicle", "MaintenanceTicket", "Mechanic", "SparePart"],
            [("Vehicle", "MaintenanceTicket"), ("Mechanic", "MaintenanceTicket"), ("MaintenanceTicket", "SparePart")],
            [
                ("Which entity tracks repair history?", "MaintenanceTicket aggregates repair events for each Vehicle.", "component_identification"),
                ("How does Mechanic associate with MaintenanceTicket?", "Mechanic is assigned to execute specific MaintenanceTicket jobs (1:N).", "relationship_reasoning"),
                ("What happens if Vehicle is decommissioned?", "MaintenanceTicket history is archived for compliance.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_008", "Social Media Engagement ERD", "ER_DIAGRAM", "SIMPLE",
            ["UserAccount", "UserPost", "CommentRecord", "LikeReaction"],
            [("UserAccount", "UserPost"), ("UserPost", "CommentRecord"), ("UserPost", "LikeReaction")],
            [
                ("Which table stores reactions on posts?", "LikeReaction records user upvotes and likes on UserPost.", "component_identification"),
                ("What relationship connects UserPost and CommentRecord?", "One UserPost can have multiple CommentRecord entries (1:N).", "relationship_reasoning"),
                ("If UserPost is deleted, what is the database action?", "Child CommentRecord and LikeReaction entries are cascade deleted.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_009", "Banking Ledger Schema ERD", "ER_DIAGRAM", "SIMPLE",
            ["AccountHolder", "Account", "LedgerEntry", "BankBranch"],
            [("AccountHolder", "Account"), ("Account", "LedgerEntry"), ("Account", "BankBranch")],
            [
                ("Where are financial credits and debits logged?", "LedgerEntry stores immutable double-entry transaction rows for each Account.", "component_identification"),
                ("What connects AccountHolder to Account?", "AccountHolder owns one or more Account records (1:N).", "relationship_reasoning"),
                ("What happens if Account is frozen?", "New LedgerEntry transactions cannot be posted to the frozen account.", "failure_impact")
            ]
        ),
        (
            "flowchart_er_010", "Warehouse Logistics ERD", "ER_DIAGRAM", "SIMPLE",
            ["WarehouseFacility", "StorageAisle", "PalletLocation", "InventorySKU"],
            [("WarehouseFacility", "StorageAisle"), ("StorageAisle", "PalletLocation"), ("PalletLocation", "InventorySKU")],
            [
                ("What hierarchical structure models warehouse storage?", "WarehouseFacility contains StorageAisles, which contain PalletLocations holding InventorySKUs.", "structural_analysis"),
                ("Identify the terminal item table.", "InventorySKU defines product catalog items stored in physical locations.", "component_identification"),
                ("If a StorageAisle is closed, what is affected?", "All PalletLocations and InventorySKUs within that aisle are inaccessible.", "failure_impact")
            ]
        ),
        # Control Flowcharts (11-20)
        (
            "flowchart_er_011", "User Authentication & 2FA Flowchart", "FLOWCHART", "MEDIUM",
            ["Start", "InputCredentials", "ValidatePassword", "Check2FAEnabled", "PromptOTP", "LoginSuccess", "LockoutAccount"],
            [("Start", "InputCredentials"), ("InputCredentials", "ValidatePassword"), ("ValidatePassword", "Check2FAEnabled"), ("ValidatePassword", "LockoutAccount"), ("Check2FAEnabled", "PromptOTP"), ("Check2FAEnabled", "LoginSuccess"), ("PromptOTP", "LoginSuccess"), ("PromptOTP", "LockoutAccount")],
            [
                ("What decision determines whether OTP is requested?", "Check2FAEnabled checks if multi-factor authentication is active on the user account.", "structural_analysis"),
                ("Under what condition is LockoutAccount triggered?", "When ValidatePassword fails exceeding maximum retry limits or when OTP verification fails.", "failure_impact"),
                ("Trace the happy path for a user without 2FA.", "Start -> InputCredentials -> ValidatePassword (Valid) -> Check2FAEnabled (No) -> LoginSuccess.", "flow_tracing")
            ]
        ),
        (
            "flowchart_er_012", "E-Commerce Order Fulfillment Flowchart", "FLOWCHART", "MEDIUM",
            ["OrderPlaced", "CheckInventory", "ProcessPayment", "PickAndPack", "DispatchCourier", "CancelOrder"],
            [("OrderPlaced", "CheckInventory"), ("CheckInventory", "ProcessPayment"), ("CheckInventory", "CancelOrder"), ("ProcessPayment", "PickAndPack"), ("ProcessPayment", "CancelOrder"), ("PickAndPack", "DispatchCourier")],
            [
                ("What happens if inventory check fails?", "The workflow branches from CheckInventory directly to CancelOrder.", "flow_tracing"),
                ("Which step precedes PickAndPack?", "ProcessPayment must successfully settle funds before warehouse staff picks and packs items.", "relationship_reasoning"),
                ("Trace the terminal completion node.", "DispatchCourier handoffs the package to shipping, completing fulfillment.", "component_identification")
            ]
        ),
        (
            "flowchart_er_013", "Support Ticket Escalation Flowchart", "FLOWCHART", "MEDIUM",
            ["TicketLogged", "Tier1Triage", "CheckResolved", "Tier2Escalation", "ManagerReview", "CloseTicket"],
            [("TicketLogged", "Tier1Triage"), ("Tier1Triage", "CheckResolved"), ("CheckResolved", "CloseTicket"), ("CheckResolved", "Tier2Escalation"), ("Tier2Escalation", "ManagerReview"), ("ManagerReview", "CloseTicket")],
            [
                ("When does a ticket escalate to Tier2?", "When CheckResolved evaluated by Tier1 triage is negative (issue unresolved).", "flow_tracing"),
                ("What is the final step before ticket closure on complex bugs?", "ManagerReview provides final authorization before branching to CloseTicket.", "component_identification"),
                ("What happens if Tier1 resolves the issue immediately?", "CheckResolved branches directly to CloseTicket, bypassing higher tiers.", "structural_analysis")
            ]
        ),
        (
            "flowchart_er_014", "Loan Credit Approval Decision Flowchart", "FLOWCHART", "MEDIUM",
            ["ApplyLoan", "CreditScoreCheck", "DebtIncomeRatio", "CollateralAudit", "ApproveLoan", "RejectLoan"],
            [("ApplyLoan", "CreditScoreCheck"), ("CreditScoreCheck", "DebtIncomeRatio"), ("CreditScoreCheck", "RejectLoan"), ("DebtIncomeRatio", "CollateralAudit"), ("DebtIncomeRatio", "RejectLoan"), ("CollateralAudit", "ApproveLoan"), ("CollateralAudit", "RejectLoan")],
            [
                ("What is the first risk gate in loan processing?", "CreditScoreCheck filters applicants below minimum risk threshold.", "component_identification"),
                ("What conditions are necessary for ApproveLoan?", "Passing CreditScoreCheck, satisfying DebtIncomeRatio, and passing CollateralAudit.", "structural_analysis"),
                ("Where does the process exit on failing DTI ratio?", "DebtIncomeRatio branches directly to RejectLoan.", "flow_tracing")
            ]
        ),
        (
            "flowchart_er_015", "CI/CD Automated Deployment Flowchart", "FLOWCHART", "MEDIUM",
            ["CodePush", "RunUnitTests", "BuildDockerImage", "DeployStaging", "RunIntegrationTests", "DeployProduction", "AlertDevs"],
            [("CodePush", "RunUnitTests"), ("RunUnitTests", "BuildDockerImage"), ("RunUnitTests", "AlertDevs"), ("BuildDockerImage", "DeployStaging"), ("DeployStaging", "RunIntegrationTests"), ("RunIntegrationTests", "DeployProduction"), ("RunIntegrationTests", "AlertDevs")],
            [
                ("What triggers AlertDevs?", "Failure during RunUnitTests or failure during staging RunIntegrationTests.", "failure_impact"),
                ("What prerequisite must pass before DeployProduction?", "RunIntegrationTests must pass on DeployStaging environment.", "structural_analysis"),
                ("Trace the successful build path.", "CodePush -> RunUnitTests -> BuildDockerImage -> DeployStaging -> RunIntegrationTests -> DeployProduction.", "flow_tracing")
            ]
        ),
        (
            "flowchart_er_016", "Software Bug Triage Flowchart", "FLOWCHART", "SIMPLE",
            ["BugReported", "ReproduceBug", "IsCritical", "HotfixDeploy", "SprintBacklog", "DiscardReport"],
            [("BugReported", "ReproduceBug"), ("ReproduceBug", "IsCritical"), ("ReproduceBug", "DiscardReport"), ("IsCritical", "HotfixDeploy"), ("IsCritical", "SprintBacklog")],
            [
                ("When is a report discarded?", "When ReproduceBug fails to recreate the defect.", "flow_tracing"),
                ("What action is taken for critical bugs?", "IsCritical branches immediately to HotfixDeploy.", "structural_analysis"),
                ("Where do non-critical bugs go?", "They are queued into SprintBacklog for normal scheduling.", "component_identification")
            ]
        ),
        (
            "flowchart_er_017", "IoT Firmware OTA Update Flowchart", "FLOWCHART", "SIMPLE",
            ["CheckVersion", "BatteryCheck", "DownloadPayload", "FlashFirmware", "RebootVerify", "AbortUpdate"],
            [("CheckVersion", "BatteryCheck"), ("BatteryCheck", "DownloadPayload"), ("BatteryCheck", "AbortUpdate"), ("DownloadPayload", "FlashFirmware"), ("FlashFirmware", "RebootVerify"), ("RebootVerify", "AbortUpdate")],
            [
                ("Why is BatteryCheck necessary before download?", "To prevent bricking the IoT device if power is lost during firmware flashing.", "structural_analysis"),
                ("What happens if RebootVerify fails?", "The device rolls back and branches to AbortUpdate.", "failure_impact"),
                ("Trace the full update path.", "CheckVersion -> BatteryCheck -> DownloadPayload -> FlashFirmware -> RebootVerify.", "flow_tracing")
            ]
        ),
        (
            "flowchart_er_018", "ATM Cash Withdrawal Flowchart", "FLOWCHART", "SIMPLE",
            ["CardInserted", "VerifyPIN", "CheckBalance", "DispenseCash", "EjectCard", "RetainCard"],
            [("CardInserted", "VerifyPIN"), ("VerifyPIN", "CheckBalance"), ("VerifyPIN", "RetainCard"), ("CheckBalance", "DispenseCash"), ("CheckBalance", "EjectCard"), ("DispenseCash", "EjectCard")],
            [
                ("Under what condition is RetainCard triggered?", "When VerifyPIN fails exceeding maximum 3 PIN attempts.", "failure_impact"),
                ("When does DispenseCash execute?", "When VerifyPIN succeeds and CheckBalance confirms sufficient account funds.", "structural_analysis"),
                ("What is the final exit step?", "EjectCard returns the card safely to the customer.", "flow_tracing")
            ]
        ),
        (
            "flowchart_er_019", "Machine Learning Training Pipeline Flowchart", "FLOWCHART", "SIMPLE",
            ["IngestDataset", "CleanFeatures", "TrainModel", "EvaluateAccuracy", "DeployEndpoint", "RetrainHyperparams"],
            [("IngestDataset", "CleanFeatures"), ("CleanFeatures", "TrainModel"), ("TrainModel", "EvaluateAccuracy"), ("EvaluateAccuracy", "DeployEndpoint"), ("EvaluateAccuracy", "RetrainHyperparams"), ("RetrainHyperparams", "TrainModel")],
            [
                ("Identify the cyclical feedback loop in this flowchart.", "EvaluateAccuracy loops back to RetrainHyperparams and TrainModel if accuracy is below threshold.", "structural_analysis"),
                ("What condition allows DeployEndpoint?", "When EvaluateAccuracy meets or exceeds the required target validation score.", "flow_tracing"),
                ("What step precedes TrainModel?", "CleanFeatures cleans and normalizes raw dataset features.", "component_identification")
            ]
        ),
        (
            "flowchart_er_020", "Customer Refund Processing Flowchart", "FLOWCHART", "SIMPLE",
            ["RequestRefund", "Validate30Days", "InspectReturn", "IssueRefund", "DenyRefund"],
            [("RequestRefund", "Validate30Days"), ("Validate30Days", "InspectReturn"), ("Validate30Days", "DenyRefund"), ("InspectReturn", "IssueRefund"), ("InspectReturn", "DenyRefund")],
            [
                ("What is the initial validation gate?", "Validate30Days verifies if the purchase was made within the 30-day return policy.", "component_identification"),
                ("What causes DenyRefund?", "Exceeding 30 days or failing physical inspection during InspectReturn.", "failure_impact"),
                ("Trace the successful refund path.", "RequestRefund -> Validate30Days -> InspectReturn -> IssueRefund.", "flow_tracing")
            ]
        )
    ]

    for fid, name, cat, comp, nodes, edges, qa_tuples in er_flowchart_configs:
        node_objs = [{"id": n, "type": "entity" if cat == "ER_DIAGRAM" else ("decision" if "Check" in n or "Validate" in n or "Is" in n or "Evaluate" in n else "process")} for n in nodes]
        edge_objs = [{"from": u, "to": v, "type": "relates_to" if cat == "ER_DIAGRAM" else "transitions_to"} for u, v in edges]
        benchmarks.append({
            "diagram_id": fid,
            "title": name,
            "category": cat,
            "complexity": comp,
            "nodes": node_objs,
            "edges": edge_objs,
            "qa_pairs": [
                {"question": q, "reference_answer": a, "question_type": t} for q, a, t in qa_tuples
            ]
        })

    return benchmarks


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-RESOLUTION IMAGE & GRAPH RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def render_diagram_image(bench, output_png_path):
    """
    Renders a high-resolution, crisp diagram image with clean node boxes,
    labels, and directed relationship arrows. Computes exact pixel bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor('#0f172a')  # Dark slate modern canvas
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 700)
    ax.axis('off')

    # Draw Title Header
    cat_colors = {
        "UML_CLASS": "#38bdf8",      # Sky blue
        "SW_ARCHITECTURE": "#a855f7",# Purple
        "UML_SEQUENCE": "#34d399",   # Emerald
        "ER_DIAGRAM": "#f59e0b",     # Amber
        "FLOWCHART": "#ec4899"       # Pink
    }
    accent = cat_colors.get(bench["category"], "#38bdf8")

    plt.text(40, 660, bench["title"], fontsize=14, fontweight='bold', color='#f8fafc', fontfamily='sans-serif')
    plt.text(40, 638, f"Category: {bench['category']} | Complexity: {bench['complexity']} | ID: {bench['diagram_id']}", fontsize=9, color=accent, fontfamily='sans-serif')

    nodes = bench["nodes"]
    edges = bench["edges"]
    num_nodes = len(nodes)

    # Layout nodes in a clean topological grid or circle
    node_coords = {}
    node_bboxes = {}

    box_w = 160
    box_h = 60

    if num_nodes <= 5:
        # Single or 2-row layout
        cols = num_nodes if num_nodes <= 4 else 3
        for i, node in enumerate(nodes):
            r = i // cols
            c = i % cols
            x = 80 + c * (800 // max(1, cols))
            y = 480 - r * 180
            node_coords[node["id"]] = (x + box_w // 2, y + box_h // 2)
            node_bboxes[node["id"]] = [int(x), int(y), int(x + box_w), int(y + box_h)]
    elif num_nodes <= 8:
        cols = 4
        for i, node in enumerate(nodes):
            r = i // cols
            c = i % cols
            x = 60 + c * 220
            y = 480 - r * 180
            node_coords[node["id"]] = (x + box_w // 2, y + box_h // 2)
            node_bboxes[node["id"]] = [int(x), int(y), int(x + box_w), int(y + box_h)]
    else:
        cols = 5
        for i, node in enumerate(nodes):
            r = i // cols
            c = i % cols
            x = 40 + c * 180
            y = 500 - r * 130
            node_coords[node["id"]] = (x + 70, y + 25)
            node_bboxes[node["id"]] = [int(x), int(y), int(x + 140), int(y + 50)]

    # Draw Edges (Arrows)
    for edge in edges:
        u = edge["from"]
        v = edge["to"]
        if u in node_coords and v in node_coords:
            ux, uy = node_coords[u]
            vx, vy = node_coords[v]
            
            # Shorten arrow endpoints to not overlap box
            dx = vx - ux
            dy = vy - uy
            dist = max(1.0, (dx**2 + dy**2)**0.5)
            shrink_start = 35.0 / dist
            shrink_end = 45.0 / dist
            
            sx = ux + dx * shrink_start
            sy = uy + dy * shrink_start
            ex = vx - dx * shrink_end
            ey = vy - dy * shrink_end

            arrow = patches.FancyArrowPatch(
                (sx, sy), (ex, ey),
                arrowstyle='-|>',
                mutation_scale=14,
                color='#94a3b8',
                linewidth=1.8,
                linestyle='solid' if edge.get("type") != "inherits" else 'dashed',
                zorder=2
            )
            ax.add_patch(arrow)

    # Draw Node Boxes & Labels
    for node in nodes:
        nid = node["id"]
        if nid in node_bboxes:
            bx1, by1, bx2, by2 = node_bboxes[nid]
            bw = bx2 - bx1
            bh = by2 - by1

            # Rounded Box
            rect = patches.FancyBboxPatch(
                (bx1, by1), bw, bh,
                boxstyle="round,pad=3,rounding_size=6",
                facecolor='#1e293b',
                edgecolor=accent,
                linewidth=1.8,
                zorder=3
            )
            ax.add_patch(rect)

            # Node Name
            plt.text(
                bx1 + bw / 2, by1 + bh / 2 + 6,
                nid,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='#f8fafc',
                fontfamily='sans-serif',
                zorder=4
            )
            # Node Subtitle / Type
            ntype = node.get("type", "entity")
            plt.text(
                bx1 + bw / 2, by1 + bh / 2 - 12,
                f"«{ntype}»",
                ha='center', va='center',
                fontsize=7,
                color='#94a3b8',
                fontfamily='sans-serif',
                zorder=4
            )

    plt.tight_layout()
    plt.savefig(str(output_png_path), format='png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    return node_bboxes


def generate_plantuml_source(bench):
    """Generates standard PlantUML text for a benchmark diagram."""
    cat = bench["category"]
    puml = ["@startuml", f"' Diagram: {bench['title']}", f"' Category: {cat}"]
    puml.append("skinparam backgroundColor #0f172a")
    puml.append("skinparam defaultFontColor #f8fafc")
    puml.append("skinparam classBackgroundColor #1e293b")
    puml.append("skinparam classBorderColor #38bdf8")

    if cat == "UML_CLASS":
        for node in bench["nodes"]:
            puml.append(f"class {node['id']} {{")
            for a in node.get("attrs", []):
                puml.append(f"  {a}")
            for m in node.get("methods", []):
                puml.append(f"  {m}")
            puml.append("}")
        for edge in bench["edges"]:
            puml.append(f"{edge['from']} --> {edge['to']} : {edge.get('type', 'associates')}")

    elif cat == "SW_ARCHITECTURE":
        for node in bench["nodes"]:
            puml.append(f"node \"{node['id']}\" as {node['id']}")
        for edge in bench["edges"]:
            puml.append(f"{edge['from']} --> {edge['to']} : {edge.get('type', 'calls')}")

    elif cat == "UML_SEQUENCE":
        for node in bench["nodes"]:
            puml.append(f"participant \"{node['id']}\" as {node['id']}")
        for edge in bench["edges"]:
            puml.append(f"{edge['from']} -> {edge['to']} : {edge.get('type', 'message')}")

    else:
        for node in bench["nodes"]:
            puml.append(f"rectangle \"{node['id']}\" as {node['id']}")
        for edge in bench["edges"]:
            puml.append(f"{edge['from']} --> {edge['to']} : {edge.get('type', 'step')}")

    puml.append("@enduml\n")
    return "\n".join(puml)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_full_benchmark_dataset():
    print("\n" + "=" * 80)
    print("[*] BUILDING DIAGRAMIQ-EVAL 100-DIAGRAM REAL-WORLD RESEARCH BENCHMARK")
    print("=" * 80)

    benchmarks = get_all_benchmark_definitions()
    print(f"[*] Total benchmark diagram specifications defined: {len(benchmarks)}")

    qa_records = []
    
    cat_counts = {}

    for idx, bench in enumerate(benchmarks, 1):
        did = bench["diagram_id"]
        cat = bench["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # 1. Render PNG image
        png_path = IMAGES_DIR / f"{did}.png"
        node_bboxes = render_diagram_image(bench, png_path)

        # 2. Write PUML source
        puml_path = PUML_DIR / f"{did}.puml"
        puml_content = generate_plantuml_source(bench)
        with open(puml_path, "w", encoding="utf-8") as f:
            f.write(puml_content)

        # 3. Write Ground Truth NetworkX Graph JSON
        gt_graph_path = GT_GRAPHS_DIR / f"{did}.json"
        gt_data = {
            "diagram_id": did,
            "title": bench["title"],
            "category": cat,
            "complexity": bench["complexity"],
            "nodes": [
                {
                    "id": n["id"],
                    "type": n.get("type", "entity"),
                    "bounding_box": node_bboxes.get(n["id"], [0, 0, 100, 100]),
                    "attributes": n.get("attrs", []),
                    "methods": n.get("methods", [])
                }
                for n in bench["nodes"]
            ],
            "edges": bench["edges"],
            "image_file": f"images/{did}.png",
            "puml_file": f"plantuml_sources/{did}.puml"
        }
        with open(gt_graph_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=2)

        # 4. Collect 3 QA pairs per diagram
        for qa in bench["qa_pairs"]:
            qa_records.append({
                "diagram_id": did,
                "diagram_title": bench["title"],
                "diagram_type": cat,
                "complexity": bench["complexity"],
                "question": qa["question"],
                "reference_answer": qa["reference_answer"],
                "question_type": qa["question_type"],
                "image_path": f"images/{did}.png"
            })

        if idx % 10 == 0 or idx == len(benchmarks):
            print(f"    [+] Processed {idx:03d}/{len(benchmarks)} diagrams ({did} - {bench['title']})")

    # 5. Write Master QA CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["diagram_id", "diagram_title", "diagram_type", "complexity", "question", "reference_answer", "question_type", "image_path"]
        )
        writer.writeheader()
        writer.writerows(qa_records)

    print("\n" + "=" * 80)
    print("[SUCCESS] DiagramIQ-Eval Benchmark Dataset Built Successfully!")
    print(f"  • Total Diagrams Generated: {len(benchmarks)}")
    print(f"  • Category Breakdown: {cat_counts}")
    print(f"  • Rendered PNG Images: {len(list(IMAGES_DIR.glob('*.png')))} in {IMAGES_DIR}")
    print(f"  • Ground-Truth JSON Graphs: {len(list(GT_GRAPHS_DIR.glob('*.json')))} in {GT_GRAPHS_DIR}")
    print(f"  • PlantUML Sources: {len(list(PUML_DIR.glob('*.puml')))} in {PUML_DIR}")
    print(f"  • Master Ground-Truth QA Pairs: {len(qa_records)} in {CSV_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    build_full_benchmark_dataset()
