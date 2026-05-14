Create Databricks App to host custom MCP server (repo here)

just deploy the root 


Create service credential using an IAM role that has read access to where raw spark logs are stored/ access to certain EMR APIs to interact with persistent APP UI (further instructions in ReadMe here)

create iam role using terraform
create service cred using terraform
