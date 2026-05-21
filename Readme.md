# creating project
1. create project
2. create git

# make sure to have installed git ( version control software )

# how to connect with Git when starting project
1.install.packages("gitcreds")
2.library(gitcreds)
3.install.packages("usethis")
4.library(usethis)
5.go to github > settings > developer settings (at the bottom left)
6.generate personal access token class
7.select repo, user
8.generate 
9.gitcreds_set()
10.paste it in
11.go to tools -> version control -> commit
12.push

# how to pull from remote/github
1.new project
2.version control 
3. go to github, copy the repository URL ( click the green button with <> Code top right )

# daily routine before starting to write code, 
1.tools > version control > pull 

# done writting code?
1. tools > version control > commit > write message > commit > push 
