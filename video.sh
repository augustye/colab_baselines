for file in ./_record/*.bk2
do
	echo "$file"
	python -m retro.scripts.playback_movie "$file"
done
