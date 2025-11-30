class ModelComparator:
    """
    Compares machine learning model performance across various metrics.

    This class demonstrates several core object-oriented programming (OOP) principles:
     1. Encapsulation: The model results are stored in a protected attribute (_models_results).
     2. Operator Overloading (Dunder Methods): Customizing behavior for operators like == (equality) and > (greater than).

    Attributes:
        _models_results (Dict[str, Dict[str, float]]): The raw input dictionary
            where keys are model names and values are dictionaries of {metric: score}.
        _df (pd.DataFrame): A pandas DataFrame representation of the results,
            with models as the index and metrics as the columns.
    """
    
    def __init__(self, models_results: Dict[str, Dict[str, float]]):
        """
        Initializes the ModelComparator with a dictionary of model results.

        Args:
            models_results: A dictionary containing model performance scores.
                            Format: {'Model_A': {'Metric1': 0.9, 'Metric2': 0.8}, ...}
        """
        # Protected attribute to store the raw results
        self._models_results = models_results  
        # Convert results to a DataFrame for easier processing (models as index, metrics as columns)
        self._df = pd.DataFrame(models_results).T
    
    def create_bar_chart(self, ax, df: pd.DataFrame):
        """
        Creates and formats a grouped bar chart for model performance.

        Visualizes the performance scores, grouping bars by metric for each model.

        Args:
            ax: A Matplotlib Axes object to draw the plot on.
            df: The performance DataFrame to plot.
        """
        df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_title('Model Performance Comparison (Bar Chart)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        
        # Add score labels on top of each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    def create_radar_chart(self, ax, df: pd.DataFrame):
        """
        Creates and formats a radar  chart for model performance.

        This chart is useful for comparing models based on multiple metrics simultaneously,
        showing which model 'dominates' or balances performance across all axes.

        Args:
            ax: A Matplotlib Polar Axes object (must be created with projection='polar').
            df: The performance DataFrame  to plot.
        """
        n_metrics = len(df.columns)
        # Calculate angles for the radar axes
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        # Complete the loop for plotting
        angles += angles[:1]
        
        # Plot each model's performance on the radar chart
        for model_name, values in df.iterrows():
            values_list = values.tolist()
            values_list += values_list[:1]
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values_list, alpha=0.15)
        
        # Set the labels for each metric
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df.columns, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title('Model Performance Radar', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    def compare_models(self, metrics: Optional[List[str]] = None):
        """
        Generates and displays two visualizations 
        comparing the performance of the models.

        If `metrics` is provided, only those metrics will be plotted.

        Args:
            metrics: Optional list of specific metric names to include
                     in the visualization. If None, all available metrics are used.
        """
        if not self._models_results:
            print("Error: No model results provided")
            return
        
        df = self._df.copy()
        if metrics:
            df = df[metrics]
        
        # Create a figure with two subplots side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar chart on the left
        self.create_bar_chart(axes[0], df)
        
        # Radar chart on the right (requires 'polar' projection)
        ax_radar = plt.subplot(1, 2, 2, projection='polar')
        self.create_radar_chart(ax_radar, df)
        
        plt.tight_layout()
        plt.show() 
        
        print("\nModel Performance Summary:")
        print("="*60)
        print(df.to_string())
        print("="*60)
    
    def get_best_model(self, metric: str) -> str:
        """
        Retrieves the name of the model with the highest score for a specified metric.

        Args:
            metric: The name of the performance metric.

        Returns:
            The string name of the best-performing model for that metric.

        Raises:
            ValueError: If the specified metric is not present in the model results.
        """
        if metric not in self._df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(self._df.columns)}")
        # idxmax() returns the index (model name) of the maximum value in the specified column
        return self._df[metric].idxmax()
    
    # --- Dunder Methods (Operator Overloading) ---
    
    def __repr__(self) -> str:
        """
        Returns a developer-friendly, official string representation of the object.
        """
        return f"ModelComparator(models={len(self._models_results)}, metrics={list(self._df.columns)})"
    
    def __eq__(self, other) -> bool:
        """
        Implements the equality operator (==) for ModelComparator objects.

        Two ModelComparator objects are considered equal if they are of the same
        type and their underlying performance DataFrames contain identical data.

        Args:
            other: The object to compare against.

        Returns:
            True if the objects are equal in value and type, False otherwise.
        """
        if not isinstance(other, ModelComparator):
            return False
        # pd.DataFrame.equals() performs a rigorous comparison of data and index/column names
        return self._df.equals(other._df)
    
    def __gt__(self, other) -> bool:
        """
        Implements the greater than operator (>) for ModelComparator objects.

        Comparison is based on the average overall performance, which is calculated
        as the mean of all metric means across all models in each object's dataset.
        (i.e., mean of all scores).

        Args:
            other: The ModelComparator object to compare against.

        Returns:
            True if this object's mean performance is strictly greater than the other's.
        """
        if not isinstance(other, ModelComparator):
            return NotImplemented
        # Calculate the mean of all metrics for all models, then take the mean of those means
        return self._df.mean().mean() > other._df.mean().mean()
