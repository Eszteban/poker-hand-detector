class Card:
    def __init__(self, rank="Unknown", suit="Unknown"):
        self.rank = rank
        self.suit = suit

    def isValidCard(self):
        return self.rank != "Unknown" and self.suit != "Unknown"

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


class CardPack:
    def __init__(self):
        self.cards = []

    def addCard(self, card):
        if isinstance(card, Card):
            self.cards.append(card)
            return True
        else:
            print("Only Card objects can be added!")
            return False

    def checkHand(self):
        if len(self.cards) != 5:
            return None

        ranks = [card.rank for card in self.cards]
        suits = [card.suit for card in self.cards]

        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }

        try:
            values = sorted([rank_values[r] for r in ranks])
        except KeyError:
            return "High card"

        rank_count = {r: ranks.count(r) for r in ranks}
        counts = list(rank_count.values())
        unique_ranks = len(rank_count)
        is_flush = len(set(suits)) == 1
        is_straight = all(values[i] + 1 == values[i + 1] for i in range(4))

        # Royal flush: 10, J, Q, K, A, same suit
        if is_flush and set(values) == {10, 11, 12, 13, 14}:
            return "Royal flush"

        # Straight flush: five consecutive cards of the same suit
        if is_straight and is_flush:
            return "Straight flush"

        # Four of a kind
        if 4 in counts:
            return "Four of a kind"

        # Full house: three of a kind + a pair
        if sorted(counts) == [2, 3]:
            return "Full house"

        # Flush: all cards of the same suit
        if is_flush:
            return "Flush"

        # Straight: five consecutive cards of different suits
        if is_straight:
            return "Straight"

        # Three of a kind
        if 3 in counts:
            return "Three of a kind"

        # Two pair
        if counts.count(2) == 2:
            return "Two pair"

        # One pair
        if 2 in counts:
            return "One pair"

        return "High card"
