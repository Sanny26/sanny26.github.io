#!/bin/bash

# Check if the user has provided the resume path
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_resume>"
  exit 1
fi

# Get the resume path from the first argument
resume_path=$1

# Check if the file exists
if [ ! -f "$resume_path" ]; then
  echo "Error: File '$resume_path' not found!"
  exit 1
fi

# Ask the user whether it's for research or engineering
echo "Is this for research or engineering? (Enter 'research' or 'eng')"
read category

# Determine the target filename
if [ "$category" == "research" ]; then
  target="SG_research.pdf"
elif [ "$category" == "eng" ]; then
  target="SG_eng.pdf"
else
  echo "Invalid input. Please enter 'research' or 'eng'."
  exit 1
fi

# Define the destination path
destination="static/pdfs/$target"

# Copy the file to the static folder and rename it
cp "$resume_path" "$destination"

# Check if the copy was successful
if [ $? -eq 0 ]; then
  echo "Resume successfully copied and renamed to '$destination'"
else
  echo "Error: Failed to copy the resume."
  exit 1
fi
