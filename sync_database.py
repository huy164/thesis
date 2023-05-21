from google.cloud import firestore

def sync_documents(source_db, dest_db):
    source_collection_ref = source_db.collection('prediction')
    dest_collection_ref = dest_db.collection('prediction')

    # Retrieve all documents from the source collection
    source_docs = source_collection_ref.get()

    # Retrieve all document IDs from the destination collection
    dest_doc_ids = [doc.id for doc in dest_collection_ref.get()]

    for source_doc in source_docs:
        doc_id = source_doc.id

        # Check if the document has already been synced
        if doc_id in dest_doc_ids:
            print(f"Document {doc_id} already exists in the destination database. Skipping...")
            continue

        # Write the document to the destination collection
        dest_doc_ref = dest_collection_ref.document(doc_id)
        dest_doc_ref.set(source_doc.to_dict())
    
    
    print(f"Document {doc_id} synced successfully.")

# Initialize connections to both Firestore databases
db1 = firestore.Client.from_service_account_json('firebase-model-key.json')
db2 = firestore.Client.from_service_account_json('firebase-api-key.json')

# Sync documents from Database 1 to Database 2
sync_documents(db1, db2)
