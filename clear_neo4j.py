import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def clear_database():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    pwd = os.environ.get("NEO4J_PASSWORD", "password")
    db_name = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session(database=db_name) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("[Neo4j Cleaned] Deleted all stale nodes and relationships from previous video runs.")
        driver.close()
    except Exception as e:
        print(f"[Neo4j Cleanup Warning] Failed to clear database: {e}")

if __name__ == "__main__":
    clear_database()
