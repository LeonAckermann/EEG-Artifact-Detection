#!/bin/bash

# Get the IP address from the first argument
IP_ADDRESS="$1"

# Check if the IP address is empty
if [[ -z "$IP_ADDRESS" ]]; then
  echo "Error: Please provide an IP address"
  exit 1
fi

# Use the IP address in the scp command
scp -r -P 10200 "root@$IP_ADDRESS:./logs" ./Desktop
