#!/bin/bash

# Ask the user for a commit message
echo "Enter commit message: "
read commit_message

# Add all changed files to the staging area
git add .

# Commit the changes
git commit -m "$commit_message"

# Push the changes
git push 
