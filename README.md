# cnn-latex-character-recognition
Sparse CNN implementation for LaTeX character recognition


# Importing the data to MongoDB

SQL -> POSTGRES -> CSV -> MONGODB [ Python ]

General command
gunzip -c detexify.sql.gz | psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER detexify 

My command:
gunzip -c detexify.sql.gz | psql -h localhost -p 5432 -U samuelstewart detexify 

1. Import to Postgres

For Mac OS X, I used: http://postgresapp.com/

Then to convert in postgres, I made a new database called 'detexify' in `psql` using appropriate connection params.

`CREATE DATABSE detexify;`

`\g` # execute the query

`\i detexify.sql` # runs the command in the file

2. Export to CSV

Then inside `psql` I ran:

`COPY samples TO '/Applications/detexify.csv' DELIMITER ',' CSV HEADER`

bash:

`mv /Applications/detexify.csv .
rm detexify.sql
`

# Representing Data

In the long run, we'll need to turn the strokes into images.

Data structure for MongoDB. Each sample should be

{
	classified_latex_code: '...',
	strokes : [
		[x_1, y_1, t_1], ...
	]
}