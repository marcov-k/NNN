#include "pch.h"
#include "Tensor.h"

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
	if (_last_gen == _forward_gen) return;

	_op_index = 0;

	for (std::weak_ptr<Tensor> r : _results)
	{
		std::shared_ptr<Tensor> result = r.lock();
		if (result)
		{
			result->clear_graph();
		}
	}

	_last_gen = _forward_gen;
}

void Tensor::finalize_forward()
{
	if (_op_index < (int)_results.size())
	{
		_results.resize(_op_index);
	}
}

void Tensor::build_topo(std::shared_ptr<Tensor> t, std::vector<std::shared_ptr<Tensor>>& topo,
	std::unordered_set<std::shared_ptr<Tensor>, TensorPtrHash, TensorPtrEqual>& visited)
{
	if (visited.contains(t)) return;
	visited.insert(t);

	for (std::shared_ptr<Tensor>& p : t->_parents)
	{
		build_topo(p, topo, visited);
	}

	topo.push_back(t);
}

void Tensor::backward()
{
	if (!_topo.has_value()) _topo.emplace();
	_topo->clear();

	if (!_visited.has_value()) _visited.emplace();
	_visited->clear();

	build_topo(shared_from_this(), *_topo, *_visited);

	for (std::shared_ptr<Tensor>& t : *_topo)
	{
		t->restore_grad();
	}

	_grad.assign(_grad.size(), 1.0);

	for (int i = (int)_topo.value().size() - 1; i >= 0; i--)
	{
		_topo.value()[i]->_backward();
	}

	for (std::shared_ptr<Tensor>& t : *_topo)
	{
		t->finalize_forward();
	}
}