import inquirer

questions = [
  inquirer.List('shell',
                message="What shell do you want to use?",
                choices=['fish', 'bash']
            ),
]
config = inquirer.prompt(questions)