plot_data:
	python -m src.viz_cli --method="data"
plot_stats:
	python -m src.viz_cli --method="stats"
plot_model:
	python -m src.viz_cli --method="model"
plot_decoys:
	python -m src.viz_cli --method="decoys"

docker_run:
	docker build -t rna_torsionbert_viz .
	docker run -it -v ${PWD}/data:/app/data -v ${PWD}/img:/app/img rna_torsionbert_viz