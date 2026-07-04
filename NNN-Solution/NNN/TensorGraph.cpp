#include "pch.h"
#include "Tensor.h"

/* Autograd graph functionality */

// Restores and resets the gradient vector of the tensor.
void Tensor::restore_grad()
{
	if (requires_grad)
	{
		_grad.assign(element_count(), 0.0);
	}
}

void Tensor::clear_graph()
{
	_parents.clear();
	_backward = [] {};
}

void Tensor::begin_forward()
{
	_forward_gen++;
}

void Tensor::prepare_forward()
{
	// Clear autograd graph connections and reset operation index if not yet participated in the current forward pass

	if (_last_gen == _forward_gen) return;

	_op_index = 0;

	for (std::shared_ptr<Tensor>& t : _results)
	{
		t->clear_graph();
	}

	_last_gen = _forward_gen;
}

void Tensor::finalize_forward()
{
	// Prune any unused result tensor allocations - prevent potential memory leaks
	if (_op_index < (int)_results.size())
	{
		_results.resize(_op_index);
	}
}

void Tensor::build_topo(const std::shared_ptr<Tensor>& t, std::vector<std::shared_ptr<Tensor>>& topo,
	std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>& visited)
{
	// Skip tensor if already visited in topography
	if (visited.contains(t)) return;
	visited.insert(t);

	// Recursively build topography starting from this tensor
	for (std::shared_ptr<Tensor>& p : t->_parents)
	{
		build_topo(p, topo, visited);
	}

	topo.push_back(t);
}

void Tensor::backward()
{
	// Initialize _topo and _visited if not yet used
	if (!_topo.has_value()) _topo.emplace();

	if (!_visited.has_value()) _visited.emplace();

	auto& topo = *_topo;
	build_topo(shared_from_this(), topo, *_visited);
	_visited->clear();

	// Zero out previous gradients
	for (std::shared_ptr<Tensor>& t : topo)
	{
		t->restore_grad();
	}

	// Set initial gradient
	_grad.assign(_grad.size(), 1.0);

	// Compute upstream gradient from each node in topography
	for (int i = (int)topo.size() - 1; i >= 0; i--)
	{
		topo[i]->_backward();
	}

	// Prune unused result tensor allocations
	for (std::shared_ptr<Tensor>& t : topo)
	{
		t->finalize_forward();
	}

	_topo->clear();
}