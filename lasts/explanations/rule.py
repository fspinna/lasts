from lasts.utils import vector_to_dict


class Inequality(object):
    def __init__(self, attribute, operator, threshold, as_contained=False):
        self.attribute = attribute
        self.operator = operator
        self.threshold = threshold
        self.as_contained = as_contained

    def __str__(self):
        if not self.as_contained:
            return "%s %s %.2f" % (self.attribute, self.operator, self.threshold)
        else:
            if ">" in self.operator:
                return "%s is contained" % self.attribute
            else:
                return "%s is not-contained" % self.attribute

    def __eq__(self, other):
        return (
            self.attribute == other.attribute
            and self.operator == other.operator
            and self.threshold == other.threshold
        )

    def __hash__(self):
        return hash(str(self))


class Rule(object):
    def __init__(self, premises, consequence, labels=None):
        self.premises = premises
        self.premises_attributes = list(
            set([premise.attribute for premise in self.premises])
        )
        self.consequence = consequence
        self.labels = labels

    def _premises_str(self):
        if len(self.premises) == 0:
            return "{ }"
        return "{ %s }" % (", ".join([str(p) for p in self.premises]))

    def _consequence_str(self):
        return (
            "{ %s }" % self.consequence
            if self.labels is None
            else "{ %s }" % self.labels[self.consequence]
        )

    def __str__(self):
        return "%s --> %s" % (self._premises_str(), self._consequence_str())

    def __eq__(self, other):
        return self.premises == other.premises and self.consequence == other.consequence

    def __len__(self):
        return len(self.premises)

    def __hash__(self):
        return hash(str(self))

    def is_covered(self, x, feature_names):
        xd = vector_to_dict(x, feature_names)
        for p in self.premises:
            if p.operator == "<=" and xd[p.attribute] > p.threshold:
                return False
            elif p.operator == ">" and xd[p.attribute] <= p.threshold:
                return False
        return True


if __name__ == "__main__":
    cond = Inequality(123, "<=", 1, as_contained=True)
    print(cond)
    rule = Rule([cond, cond], 1, labels=["zero", "one"])
    print(rule)
